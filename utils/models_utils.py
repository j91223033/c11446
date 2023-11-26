import torch
import torch.nn.functional as F
import numpy as np
import tqdm
from .evaluation import AverageMeter

@torch.no_grad()
def aug_box(self, box_infos):
    box_infos = box_infos.float().to(self.device)
    bxyz = box_infos[...,:3]
    B,N = bxyz.shape[:2]
    bxyz[..., 0] = scale_to_unit_range(bxyz[..., 0]) # normed x
    bxyz[..., 1] = scale_to_unit_range(bxyz[..., 1]) # normed y
    bxyz[..., 2] = scale_to_unit_range(bxyz[..., 2]) # normed z
    # Randomly rotate if training
    if self.training:
        rotate_matrix = get_random_rotation_matrix(self.rotate_number, self.device)
        bxyz = torch.matmul(bxyz.reshape(B*N, 3), rotate_matrix).reshape(B,N,3)        
    # multi-view
    bsize = box_infos[...,3:]
    boxs=[]
    for theta in torch.Tensor([i*2.0*np.pi/self.rotate_number for i in range(self.rotate_number)]).to(self.device):
        rotate_matrix = get_rotation_matrix(theta, self.device)
        rxyz = torch.matmul(bxyz.reshape(B*N, 3),rotate_matrix).reshape(B,N,3)
        boxs.append(torch.cat([rxyz,bsize],dim=-1))
    boxs=torch.stack(boxs,dim=1)
    return boxs
    
@torch.no_grad()
    
def aug_input(self, input_points, contrast_range=(0.5, 1.5), noise_std_dev=0.02):
    input_points = input_points.float().to(self.device)
    xyz = input_points[:, :, :, :3]
    B, N, P = xyz.shape[:3]
    input_points_multiview = []
    rgb = input_points[..., 3:6].clone()
    # Randomly rotate/color_aug if training
    if self.training:
        rotate_matrix = get_random_rotation_matrix(self.rotate_number, self.device)
        xyz = torch.matmul(xyz.reshape(B*N*P, 3), rotate_matrix).reshape(B, N, P, 3)
        rgb = get_augmented_color(rgb, contrast_range, noise_std_dev, self.device) 
    # multi-view
    for theta in torch.Tensor([i*2.0*np.pi/self.rotate_number for i in range(self.rotate_number)]).to(self.device):  
        rotate_matrix = get_rotation_matrix(theta, self.device)
        rotated_xyz = torch.matmul(xyz.reshape(B*N*P, 3), rotate_matrix).reshape(B, N, P, 3)
        rotated_input_points = torch.clone(input_points)
        rotated_input_points[..., :3] = rotated_xyz
        rotated_input_points[..., 3:6] = rgb
        input_points_multiview.append(rotated_input_points)
    # Stack list of tensors into a single tensor
    input_points_multiview = torch.stack(input_points_multiview, dim=1)
    return input_points_multiview

def make_batch_keys(config, extras=None):
    """depending on the config, different data are used by the listener."""
    batch_keys = ['objects', 'tokens', 'target_pos']  # all models use these
    if extras is not None:
        batch_keys += extras

    if config.obj_cls_alpha > 0:
        batch_keys.append('class_labels')

    if config.lang_cls_alpha > 0:
        batch_keys.append('target_class')

    return batch_keys


def single_epoch_train(model, data_loader, criteria, optimizer, device, pad_idx, config, tokenizer=None,epoch=None):
    """
    :param model:
    :param data_loader:
    :param criteria: (dict) holding all modules for computing the losses.
    :param optimizer:
    :param device:
    :param pad_idx: (int)
    :param config:
    :return:
    """

    metrics = dict()  # holding the losses/accuracies
    total_loss_mtr = AverageMeter()
    referential_loss_mtr = AverageMeter()
    obj_loss_mtr = AverageMeter()
    ref_acc_mtr = AverageMeter()
    cls_acc_mtr = AverageMeter()
    post_cls_acc_mtr = AverageMeter()
    txt_acc_mtr = AverageMeter()
    logits_analyze_all = {
        'category_top1': torch.tensor([], device=device),
        'category_top3': torch.tensor([], device=device),
        'category_top5': torch.tensor([], device=device),
        'spatial_top1': torch.tensor([], device=device),
        'spatial_top3': torch.tensor([], device=device),
        'spatial_top5': torch.tensor([], device=device),
        
        'icategory_top1': torch.tensor([], device=device),
        'icategory_top3': torch.tensor([], device=device),
        'icategory_top5': torch.tensor([], device=device),
        'ispatial_top1': torch.tensor([], device=device),
        'ispatial_top3': torch.tensor([], device=device),
        'ispatial_top5': torch.tensor([], device=device),
    }

    # Set the model in training mode
    model.train()
    np.random.seed()  # call this to change the sampling of the point-clouds
    batch_keys = make_batch_keys(config)
    for batch in tqdm.tqdm(data_loader):        
        # Move data to gpu
        for k in batch_keys:
            if isinstance(batch[k],list):
                continue
            batch[k] = batch[k].to(device)

        # if config.object_encoder == 'pnet':
        #     batch['objects'] = batch['objects'].permute(0, 1, 3, 2)
        lang_tokens = tokenizer(batch['tokens'], return_tensors='pt', padding=True)
        for name in lang_tokens.data:
            lang_tokens.data[name] = lang_tokens.data[name].cuda()
        batch['lang_tokens'] = lang_tokens
        

        # Forward pass
        LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS, POST_CLASS_LOGITS, logits_analyze = model(batch, epoch)
        LOSS = LOSS.mean()
        for key in logits_analyze_all:
            logits_analyze_all[key] = torch.cat((logits_analyze_all[key], logits_analyze[key]))

        res = {}
        res['logits'] = LOGITS
        res['class_logits'] = CLASS_LOGITS
        res['post_class_logits'] = POST_CLASS_LOGITS
        res['lang_logits'] = LANG_LOGITS
        # Backward
        optimizer.zero_grad()
        LOSS.backward()
        optimizer.step()

        # Update the loss and accuracy meters
        target = batch['target_pos']
        batch_size = target.size(0)  # B x N_Objects
        total_loss_mtr.update(LOSS.item(), batch_size)

        predictions = torch.argmax(res['logits'], dim=1)
        guessed_correctly = torch.mean((predictions == target).double()).item()
        ref_acc_mtr.update(guessed_correctly, batch_size)

        if config.obj_cls_alpha > 0:
            cls_b_acc, _ = cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=pad_idx)
            cls_acc_mtr.update(cls_b_acc, batch_size)
            cls_b_acc, _ = cls_pred_stats(res['post_class_logits'], batch['class_labels'], ignore_label=pad_idx)
            post_cls_acc_mtr.update(cls_b_acc, batch_size)

        if config.lang_cls_alpha > 0:
            batch_guess = torch.argmax(res['lang_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['target_class']).double())
            txt_acc_mtr.update(cls_b_acc, batch_size)

    metrics['train_total_loss'] = total_loss_mtr.avg
    metrics['train_referential_acc'] = ref_acc_mtr.avg
    metrics['train_object_cls_acc'] = cls_acc_mtr.avg
    metrics['train_txt_cls_acc'] = txt_acc_mtr.avg
    metrics['train_post_object_cls_acc'] = post_cls_acc_mtr.avg
    return metrics, logits_analyze_all


@torch.no_grad()
def evaluate_on_dataset(model, data_loader, criteria, device, pad_idx, config, randomize=False, tokenizer=None):
    # TODO post-deadline, can we replace this func with the train + a 'phase==eval' parameter?
    metrics = dict()  # holding the losses/accuracies
    total_loss_mtr = AverageMeter()
    referential_loss_mtr = AverageMeter()
    obj_loss_mtr = AverageMeter()
    ref_acc_mtr = AverageMeter()
    cls_acc_mtr = AverageMeter()
    txt_acc_mtr = AverageMeter()
    post_cls_acc_mtr = AverageMeter()
    logits_analyze_all = {
        'category_top1': torch.tensor([], device=device),
        'category_top3': torch.tensor([], device=device),
        'category_top5': torch.tensor([], device=device),
        'spatial_top1': torch.tensor([], device=device),
        'spatial_top3': torch.tensor([], device=device),
        'spatial_top5': torch.tensor([], device=device),
        
        'icategory_top1': torch.tensor([], device=device),
        'icategory_top3': torch.tensor([], device=device),
        'icategory_top5': torch.tensor([], device=device),
        'ispatial_top1': torch.tensor([], device=device),
        'ispatial_top3': torch.tensor([], device=device),
        'ispatial_top5': torch.tensor([], device=device),
    }

    # Set the model in training mode
    model.eval()

    if randomize:
        np.random.seed()
    else:
        np.random.seed(config.random_seed)

    batch_keys = make_batch_keys(config)
    for batch in tqdm.tqdm(data_loader):
        
        # Move data to gpu
        for k in batch_keys:
            if isinstance(batch[k],list):
                continue
            batch[k] = batch[k].to(device)

        # if config.object_encoder == 'pnet':
        #     batch['objects'] = batch['objects'].permute(0, 1, 3, 2)
        lang_tokens = tokenizer(batch['tokens'], return_tensors='pt', padding=True)
        for name in lang_tokens.data:
            lang_tokens.data[name] = lang_tokens.data[name].cuda()
        batch['lang_tokens'] = lang_tokens

        # Forward pass
        LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS, POST_CLASS_LOGITS, logits_analyze = model(batch)
        LOSS = LOSS.mean()
        for key in logits_analyze_all:
            logits_analyze_all[key] = torch.cat((logits_analyze_all[key], logits_analyze[key]))
        res = {}
        res['logits'] = LOGITS
        res['class_logits'] = CLASS_LOGITS
        res['post_class_logits'] = POST_CLASS_LOGITS
        res['lang_logits'] = LANG_LOGITS

        # Update the loss and accuracy meters
        target = batch['target_pos']
        batch_size = target.size(0)  # B x N_Objects
        total_loss_mtr.update(LOSS.item(), batch_size)

        predictions = torch.argmax(res['logits'], dim=1)
        guessed_correctly = torch.mean((predictions == target).double()).item()
        ref_acc_mtr.update(guessed_correctly, batch_size)

        if config.obj_cls_alpha > 0:
            cls_b_acc, _ = cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=pad_idx)
            cls_acc_mtr.update(cls_b_acc, batch_size)
            
            cls_b_acc, _ = cls_pred_stats(res['post_class_logits'], batch['class_labels'], ignore_label=pad_idx)
            post_cls_acc_mtr.update(cls_b_acc, batch_size)

        if config.lang_cls_alpha > 0:
            batch_guess = torch.argmax(res['lang_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['target_class']).double())
            txt_acc_mtr.update(cls_b_acc, batch_size)

    metrics['test_total_loss'] = total_loss_mtr.avg
    metrics['test_referential_acc'] = ref_acc_mtr.avg
    metrics['test_object_cls_acc'] = cls_acc_mtr.avg
    metrics['test_txt_cls_acc'] = txt_acc_mtr.avg
    metrics['test_post_object_cls_acc'] = post_cls_acc_mtr.avg
    return metrics, logits_analyze_all


@torch.no_grad()
def detailed_predictions_on_dataset(model, data_loader, config, device, FOR_VISUALIZATION=True,tokenizer=None):
    model.eval()

    res = dict()
    res['guessed_correctly'] = list()
    res['confidences_probs'] = list()
    res['contrasted_objects'] = list()
    res['target_pos'] = list()
    res['context_size'] = list()
    res['guessed_correctly_among_true_class'] = list()

    batch_keys = make_batch_keys(config, extras=['context_size', 'target_class_mask'])

    if FOR_VISUALIZATION:
        res['utterance'] = list()
        res['stimulus_id'] = list()
        res['object_ids'] = list()
        res['target_object_id'] = list()
        res['distrators_pos'] = list()

    for batch in tqdm.tqdm(data_loader):
        # Move data to gpu
        for k in batch_keys:
            if isinstance(batch[k],list):
                continue
            batch[k] = batch[k].to(device)

        # if config.object_encoder == 'pnet':
        #     batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        lang_tokens = tokenizer(batch['tokens'], return_tensors='pt', padding=True)
        for name in lang_tokens.data:
            lang_tokens.data[name] = lang_tokens.data[name].cuda()
        batch['lang_tokens'] = lang_tokens

        LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS, POST_CLASS_LOGITS, logits_analyze = model(batch)
        LOSS = LOSS.mean()
        out = {}
        out['logits'] = LOGITS
        out['class_logits'] = CLASS_LOGITS
        res['post_class_logits'] = POST_CLASS_LOGITS
        out['lang_logits'] = LANG_LOGITS

        if FOR_VISUALIZATION:
            n_ex = len(out['logits'])
            c = batch['context_size']
            n_obj = out['logits'].shape[1]
            for i in range(n_ex):
                if c[i] < n_obj:
                    out['logits'][i][c[i]:] = -10e6

        predictions = torch.argmax(out['logits'], dim=1)
        res['guessed_correctly'].append((predictions == batch['target_pos']).cpu().numpy())
        res['confidences_probs'].append(F.softmax(out['logits'], dim=1).cpu().numpy())
        res['contrasted_objects'].append(batch['class_labels'].cpu().numpy())
        res['target_pos'].append(batch['target_pos'].cpu().numpy())
        res['context_size'].append(batch['context_size'].cpu().numpy())

        if FOR_VISUALIZATION:
            res['utterance'].append(batch['utterance'])
            res['stimulus_id'].append(batch['stimulus_id'])
            res['object_ids'].append(batch['object_ids'])
            res['target_object_id'].append(batch['target_object_id'])
            res['distrators_pos'].append(batch['distrators_pos'])

        # also see what would happen if you where to constraint to the target's class.
        cancellation = -1e6
        mask = batch['target_class_mask']
        out['logits'] = out['logits'].float() * mask.float() + (~mask).float() * cancellation
        predictions = torch.argmax(out['logits'], dim=1)
        res['guessed_correctly_among_true_class'].append((predictions == batch['target_pos']).cpu().numpy())

    res['guessed_correctly'] = np.hstack(res['guessed_correctly'])
    res['confidences_probs'] = np.vstack(res['confidences_probs'])
    res['contrasted_objects'] = np.vstack(res['contrasted_objects'])
    res['target_pos'] = np.hstack(res['target_pos'])
    res['context_size'] = np.hstack(res['context_size'])
    res['guessed_correctly_among_true_class'] = np.hstack(res['guessed_correctly_among_true_class'])
    return res


@torch.no_grad()
def save_predictions_for_visualization(model, data_loader, device, channel_last, seed=2020):
    """
    Return the predictions along with the scan data for further visualization
    """
    batch_keys = ['objects', 'tokens', 'class_labels', 'target_pos', 'scan', 'bboxes']

    # Set the model in eval mode
    model.eval()

    # Create table
    res_list = []

    # Fix the test random seed
    np.random.seed(seed)

    for batch in data_loader:
        # Move the batch to gpu
        for k in batch_keys:
            if len(batch[k]) > 0:
                if isinstance(batch[k],list):
                    continue
                batch[k] = batch[k].to(device)

        if not channel_last:
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward Pass
        res = model(batch)

        batch_size = batch['target_pos'].size(0)
        for i in range(batch_size):
            res_list.append({
                'scan_id': batch['scan_id'][i],
                'utterance': batch['utterance'][i],
                'target_pos': batch['target_pos'][i].cpu(),
                'confidences': res['logits'][i].cpu().numpy(),
                'bboxes': batch['objects_bboxes'][i].cpu().numpy(),
                'predicted_classes': res['class_logits'][i].argmax(dim=-1).cpu(),
                'predicted_target_pos': res['logits'][i].argmax(-1).cpu(),
                'object_ids': batch['object_ids'][i],
                'context_size': batch['context_size'][i],
                'is_easy': batch['is_easy'][i]
            })

    return res_list


def prediction_stats(logits, gt_labels):
    """ Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
    :param logits: The output of the model (predictions) of size: B x N_Objects
    :param gt_labels: The ground truth labels of size: B x 1
    :param ignore_label: The label of the padding class (to be ignored)
    :return: The mean accuracy and lists of correct and wrong predictions
    """
    predictions = logits.argmax(dim=1)
    correct_guessed = gt_labels == predictions
    assert (type(correct_guessed) == torch.Tensor)
    mean_accuracy = torch.mean(correct_guessed.double()).item()
    return mean_accuracy


@torch.no_grad()
def cls_pred_stats(logits, gt_labels, ignore_label):
    """ Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
    :param logits: The output of the model (predictions) of size: B x N_Objects x N_Classes
    :param gt_labels: The ground truth labels of size: B x N_Objects
    :param ignore_label: The label of the padding class (to be ignored)
    :return: The mean accuracy and lists of correct and wrong predictions
    """
    predictions = logits.argmax(dim=-1)  # B x N_Objects x N_Classes --> B x N_Objects
    valid_indices = gt_labels != ignore_label

    predictions = predictions[valid_indices]
    gt_labels = gt_labels[valid_indices]

    correct_guessed = gt_labels == predictions
    assert (type(correct_guessed) == torch.Tensor)

    found_samples = gt_labels[correct_guessed]
    # missed_samples = gt_labels[torch.logical_not(correct_guessed)] # TODO  - why?
    mean_accuracy = torch.mean(correct_guessed.double()).item()
    return mean_accuracy, found_samples

def get_siamese_features(net, in_features, aggregator=None):
    """ Applies a network in a siamese way, to 'each' in_feature independently
    :param net: nn.Module, Feat-Dim to new-Feat-Dim
    :param in_features: B x  N-objects x Feat-Dim
    :param aggregator, (opt, None, torch.stack, or torch.cat)
    :return: B x N-objects x new-Feat-Dim
    """
    independent_dim = 1
    n_items = in_features.size(independent_dim)
    out_features = []
    for i in range(n_items):
        out_features.append(net(in_features[:, i]))
    if aggregator is not None:
        out_features = aggregator(out_features, dim=independent_dim)
    return out_features

def save_state_dicts(checkpoint_file, epoch=None, **kwargs):
    """Save torch items with a state_dict.
    """
    checkpoint = dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    for key, value in kwargs.items():
        checkpoint[key] = value.state_dict()

    torch.save(checkpoint, checkpoint_file)

def load_state_dicts(checkpoint_file, map_location=None, **kwargs):
    """Load torch items from saved state_dictionaries.
    """
    if map_location is None:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

    for key, value in kwargs.items():
        value.load_state_dict(checkpoint[key])

    epoch = checkpoint.get('epoch')
    if epoch:
        return epoch

def get_random_rotation_matrix(rotate_number, device):
    rotate_theta_arr = torch.Tensor([i*2.0*torch.pi/rotate_number for i in range(rotate_number)]).to(device)
    theta = rotate_theta_arr[torch.randint(0, rotate_number, (1,))]
    return get_rotation_matrix(theta, device)

def get_rotation_matrix(theta, device):
    rotate_matrix = torch.Tensor([[torch.cos(theta), -torch.sin(theta), 0.0],
                                [torch.sin(theta), torch.cos(theta),  0.0],
                                [0.0,           0.0,            1.0]]).to(device)
    return rotate_matrix

def get_augmented_color(rgb, contrast_range=(0.5, 1.5), noise_std_dev=0.02, device='cuda'):
    # RGB Augmentation
    contrast_factor = torch.empty(1).uniform_(contrast_range[0], contrast_range[1]).to(device)
    rgb = rgb * contrast_factor
    noise = torch.normal(mean=0., std=noise_std_dev, size=rgb.shape, device=device)
    rgb = rgb + noise
    rgb = torch.clamp(rgb, -1.0, 1.0)
    return rgb

def scale_to_unit_range(x):
    max_x = torch.max(x, dim=-1, keepdim=True).values
    min_x = torch.min(x, dim=-1, keepdim=True).values
    return x / (max_x - min_x + 1e-9)

def norm_output_scores(x, mask):
    # Suppose x is your tensor
    x_mean = x.mean(dim=1, keepdim=True)  # Compute mean across the feature dimension (N)
    x_std = x.std(dim=1, keepdim=True)  # Compute standard deviation across the feature dimension (N)
    # Standardize the clamped tensor
    x_standardized = (x - x_mean) / x_std # zero mean and unit variance
    return x_standardized

def rotation_aggregate(output):
    B, R, N, _ = output.shape
    """scaling_factors = torch.rand((B, R, 1, 1), device=self.device) * (1 - 0.33) + 0.33
    scaled_output = output * scaling_factors"""
    scaled_output = output
    return (scaled_output / R).sum(dim=1)

def batch_expansion(tensor, n):
    return tensor.unsqueeze(1).repeat(1, n, *([1] * (tensor.dim() - 1))).view(tensor.size(0) * n, *tensor.shape[1:])

def get_analyze(object_lang_logits, fusion_logits, final_logits, target_pos, utterance, scan_id):
    object_lang_logits = object_lang_logits.detach()
    fusion_logits = fusion_logits.detach()
    final_logits = final_logits.detach()
    category_top1, correct_final, topk_indices = analyze(object_lang_logits, target_pos, final_logits, k=1)
    category_top3, _, _ = analyze(object_lang_logits, target_pos, final_logits, k=3)
    category_top5, _, _ = analyze(object_lang_logits, target_pos, final_logits, k=5)    
    
    spatial_top1, correct_final, topk_indices = analyze(fusion_logits, target_pos, final_logits, k=1)
    spatial_top3, _, _ = analyze(fusion_logits, target_pos, final_logits, k=3)
    spatial_top5, _, _ = analyze(fusion_logits, target_pos, final_logits, k=5)    
    
    icategory_top1, incorrect_final,topk_indices = analyze_incorrect(object_lang_logits, target_pos, final_logits, k=1)
    icategory_top3, _, _ = analyze_incorrect(object_lang_logits, target_pos, final_logits, k=3)
    icategory_top5, _, _ = analyze_incorrect(object_lang_logits, target_pos, final_logits, k=5)
        
    ispatial_top1, incorrect_final,topk_indices = analyze_incorrect(fusion_logits, target_pos, final_logits, k=1)
    ispatial_top3, _, _ = analyze_incorrect(fusion_logits, target_pos, final_logits, k=3)
    ispatial_top5, _, _ = analyze_incorrect(fusion_logits, target_pos, final_logits, k=5)
    """"if (ispatial_top1).any().item():
        print(ispatial_top1)
        print(utterance)
        print(scan_id)
        print(incorrect_final)
        print(topk_indices) """
    
    logits_analyze = {
        'category_top1': category_top1,
        'category_top3': category_top3,
        'category_top5': category_top5,
        'spatial_top1': spatial_top1,
        'spatial_top3': spatial_top3,
        'spatial_top5': spatial_top5,
        
        'icategory_top1': icategory_top1,
        'icategory_top3': icategory_top3,
        'icategory_top5': icategory_top5,
        'ispatial_top1': ispatial_top1,
        'ispatial_top3': ispatial_top3,
        'ispatial_top5': ispatial_top5,
    }
    return logits_analyze
 
def analyze(logits, target_pos, final_logits, k=3):
    # Determine the batches where the final prediction is correct
    top1_indices_final = final_logits.argmax(dim=1)
    correct_final = top1_indices_final == target_pos
    # Select only the logits for the correct batches
    correct_logits = logits[correct_final]
    correct_target_pos = target_pos[correct_final]
    # Get top-k indices for each correct batch
    topk_indices = correct_logits.topk(k, dim=1)[1]
    # Compare with the correct object
    correct_in_topk = topk_indices == correct_target_pos.view(-1, 1)
    # Check if the correct object is in top-k
    correct_in_topk_any = correct_in_topk.any(dim=1)
    return correct_in_topk_any, correct_final, topk_indices

def analyze_incorrect(logits, target_pos, final_logits, k=3):
    # Determine the batches where the final prediction is incorrect
    top1_indices_final = final_logits.argmax(dim=1)
    incorrect_final = top1_indices_final != target_pos
    # Select only the logits for the incorrect batches
    incorrect_logits = logits[incorrect_final]
    incorrect_target_pos = target_pos[incorrect_final]
    # Get top-k indices for each incorrect batch
    topk_indices = incorrect_logits.topk(k, dim=1)[1]
    # Compare with the correct object
    correct_in_topk = topk_indices == incorrect_target_pos.view(-1, 1)
    # Check if the correct object is in top-k
    correct_in_topk_any = correct_in_topk.any(dim=1)
    return correct_in_topk_any, incorrect_final, topk_indices
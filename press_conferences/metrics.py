import torch
import logging



def accuracy(predictions, labels):
    """
        Predictions are a tensor of shape (3, n) of 3 predictions scores:
            hawkish, neutral, dovish.
        If label is 1, we check prediction for hawkish and compare it to the label, 
        if label is 0, we check prediction for dovish and compare it to the label.

    """
    results = []
    for i in range(len(labels)):
        if labels[i] == 1:
            results.append(predictions[0, i] > 0)
        if labels[i] == 0:
            results.append(predictions[2, i] < 0)
    return results


def main_triplet_accuracy(predictions):
    """
        Predictions are a tensor of shape (3, n) of 3 predictions scores:
            hawkish, neutral, dovish.
        We check whether hawkish > dovish.

        We check whether hawkish > dovish for an entire triplet.
    """
    accuracy_tensor = (predictions[0] > predictions[2])
    accuracy_tensor = accuracy_tensor.float()
    return accuracy_tensor.mean()


def triplet_accuracy(predictions):
    """
        Predictions are a tensor of shape (3, n) of 3 predictions scores:
            hawkish, neutral, dovish.
        We check whether hawkish > neutral > dovish.

        We check exact order for an entire triplet.
        
    """
    accuracy_tensor = (predictions[0] > predictions[1]) & (predictions[1] > predictions[2])
    accuracy_tensor = accuracy_tensor.float()
    return accuracy_tensor.mean()

def analyze_triplet_errors(predictions):
    hawkish = predictions[0]
    neutral = predictions[1]
    dovish = predictions[2]
    
    # Count different types of ordering violations
    hawkish_neutral = (hawkish <= neutral).sum().item()
    neutral_dovish = (neutral <= dovish).sum().item()
    hawkish_dovish = (hawkish <= dovish).sum().item()
    
    total_triplets = len(hawkish)

    return {
        "hawkish_neutral": hawkish_neutral,
        "neutral_dovish": neutral_dovish,
        "hawkish_dovish": hawkish_dovish,
        "total_triplets": total_triplets
    }

def pairs_accuracy(scores, labels):
    """
        Scores - tensor of shape (n) of the difference between the first and second texts in pairs
        Labels - tensor of shape (n) of labels for the pairs
    """
    diff = scores
    accurate = torch.zeros_like(labels, dtype=torch.bool)
    
    # Check conditions for each label
    # For label == 0.5 or -0.5
    mask_05 = (labels == 0.5) | (labels == -0.5)
    accurate[mask_05] = ((diff * labels.sign())[mask_05] >= 0) & ((diff * labels.sign())[mask_05] <= 1.0)
    
    # For label == 1 or -1
    mask_1 = (labels == 1) | (labels == -1)
    accurate[mask_1] = (diff * labels.sign())[mask_1] >= 1.0
    
    # For label == 0
    mask_0 = (labels == 0)
    accurate[mask_0] = (diff[mask_0] >= -0.5) & (diff[mask_0] <= 0.5)
    
    # Compute accuracy as the mean of accurate pairs
    accuracy = accurate.float().mean().item()
    
    return accuracy

def pairs_errors_decomposed(scores, labels, padding=0.25):
    """
    Analyze errors in pair scoring and print misclassified types.
    
    Args:
        scores - tensor of shape (n) of the difference between the first and second texts in pairs
        labels - tensor of shape (n) of the labels for the pairs
    """
    # Calculate the difference
    diff = scores

    #Hard Errors
    hard_05 = 0
    hard_1 = 0
    hard_0 = 0
    
    # Initialize counters for misclassified types
    soft_05 = 0
    soft_1 = 0
    soft_0 = 0
    
    # Check conditions for each label
    for i in range(len(labels)):
        if labels[i] in [0.5, -0.5]:
            if diff[i] * labels[i].sign() < 0 or diff[i] * labels[i].sign() > 1:
                hard_05 += 1
                #logging.info(f"Pair {i} with label {labels[i]} hard error: diff = {diff[i].item()}")
            else:
                if not (0.5 - padding <= diff[i] * labels[i].sign() <= 0.5 + padding):
                    soft_05 += 1
                    #logging.info(f"Pair {i} with label {labels[i]} soft error: diff = {diff[i].item()}")
        
        elif labels[i] in [1, -1]:
            if diff[i] * labels[i].sign() < 0.0:
                hard_1 += 1
                #logging.info(f"Pair {i} with label {labels[i]} hard error: diff = {diff[i].item()}")
            else:
                if not (diff[i] * labels[i].sign() > 1.0):
                    soft_1 += 1
                    #logging.info(f"Pair {i} with label {labels[i]} soft error: diff = {diff[i].item()}")
        
        elif labels[i] == 0:
            if diff[i] * labels[i].sign() > 0.5:
                hard_0 += 1
                #logging.info(f"Pair {i} with label {labels[i]} hard error: diff = {diff[i].item()}")
            if not (-padding <= diff[i] <= padding):
                soft_0 += 1
                #logging.info(f"Pair {i} with label {labels[i]} soft error: diff = {diff[i].item()}")
    
    total_05 = len([label for label in labels if label in [0.5, -0.5]])
    total_1 = len([label for label in labels if label in [1, -1]])
    total_0 = len([label for label in labels if label == 0])
    # Print summary of misclassifications
    logging.info(f"Total pairs: {len(labels)}")
    logging.info(f"Total hard errors with label 0.5 or -0.5: {hard_05} Out of {total_05}")
    logging.info(f"Total hard errors with label 1 or -1: {hard_1} Out of {total_1}")
    logging.info(f"Total hard errors with label 0: {hard_0} Out of {total_0}")

    logging.info(f"Total soft errors with label 0.5 or -0.5: {soft_05} Out of {total_05}")
    logging.info(f"Total soft errors with label 1 or -1: {soft_1} Out of {total_1}")
    logging.info(f"Total soft errors with label 0: {soft_0} Out of {total_0}")

    return {
        'hard_05': hard_05,
        'hard_1': hard_1,
        'hard_0': hard_0,
        'soft_05': soft_05,
        'soft_1': soft_1,
        'soft_0': soft_0
    }
import torch

'''def intersection_over_union(preds, targets, iou_thr):
    xA = torch.max(preds[:,0], targets[:,0])
    yA = torch.max(preds[:,1], targets[:,1])
    xB = torch.min(preds[:,2], targets[:,2])
    yB = torch.min(preds[:,3], targets[:,3])

    interArea = torch.max(torch.tensor(0.0), xB - xA + 1) * torch.max(torch.tensor(0.0), yB - yA + 1)

    boxAArea = (preds[:,2] - preds[:,0] + 1) * (preds[:,3] - preds[:,1] + 1)
    boxBArea = (targets[:,2] - targets[:,0] + 1) * (targets[:,3] - targets[:,1] + 1)
    
    iou = interArea / (boxAArea + boxBArea - interArea)        
    
    correct_predictions = (iou > iou_thr)
    return correct_predictions'''


def intersection_over_union(boxA_, boxsB, iou_thr):
    correct_predictions = []
    correct = 0
    for boxA in boxA_:
        for boxB in boxsB:
            xA = torch.max(boxA[0], boxB[0])
            yA = torch.max(boxA[1], boxB[1])
            xB = torch.min(boxA[2], boxB[2])
            yB = torch.min(boxA[3], boxB[3])

            interArea = torch.max(torch.tensor(0.0), xB - xA + 1) * torch.max(torch.tensor(0.0), yB - yA + 1)
            
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            
            iou = interArea / (boxAArea + boxBArea - interArea)
            
            if iou > iou_thr :
                correct = 1
                break  
    
        correct_predictions.append(correct)
    return torch.tensor(correct_predictions)
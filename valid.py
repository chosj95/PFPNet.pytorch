import pickle
import torch.utils.data as data
from data import *
from utils.trainutils import Timer


def _valid(model, args, iters, dataset):
    num_images = len(dataset)
    #labelmap = (COCO_CLASSES, VOC_CLASSES)['num_classes']
    labelmap = VOC_CLASSES
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]
    if not os.path.exists(os.path.join(args.eval_folder, 'valid')):
        os.mkdir(os.path.join(args.eval_folder, 'valid'))
    det_file = os.path.join(args.eval_folder, 'valid', 'detection_%d.pkl' %iters )

    data_loader = data.DataLoader(dataset, 1, num_workers=0, shuffle=False, pin_memory=False)
    model.eval()
    timer = Timer('m')
    timer.tic()
    for i, batch_data in enumerate(data_loader):
        images, gt, height, width = batch_data
        if args.cuda:
            images = images.cuda()

        with torch.no_grad():
            detections = model(images).data

        for j in range(1, detections.size(1)):

            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)

            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= width
            boxes[:, 2] *= width
            boxes[:, 1] *= height
            boxes[:, 3] *= height
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets
        print('\r%04d'%i, end='')

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    mAP = dataset.validate_detections(all_boxes, os.path.join(args.eval_folder, 'valid'))
    model.train()
    print('\r%d iters, take %.2f mins'%(iters, timer.toc()))
    return mAP
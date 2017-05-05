function roidb = roidb_from_common(experiment_dir, imdb)
% roidb = roidb_from_voc(imdb)
%   Builds an regions of interest database from imdb image
%   database. Uses precomputed selective search boxes available
%   in the R-CNN data package.
%
%   Inspired by Andrea Vedaldi's MKL imdb and roidb code.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

cache_file = [experiment_dir 'imdb/roidb_' imdb.name];
try
  load(cache_file);
catch
  VOCopts = imdb.details.VOCopts;

  addpath(fullfile(VOCopts.datadir, 'VOCcode')); 

  roidb.name = imdb.name;

  fprintf('Loading region proposals...');
  regions_file = sprintf([experiment_dir 'data/candidates/%s'], roidb.name);
  regions = load(regions_file);
  fprintf('done\n');
  for i = 1:length(imdb.image_ids)
        tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids));
        try
          voc_rec = PASreadrecord(sprintf(VOCopts.annopath,  VOCopts.testset ,imdb.image_ids{i}));
        catch
          voc_rec = [];
        end
        roidb.rois(i) = attach_proposals(voc_rec, regions.boxes{i}, imdb.class_to_id);
  end
  rmpath(fullfile(VOCopts.datadir, 'VOCcode')); 

  fprintf('Saving roidb to cache...');
  save(cache_file, 'roidb', '-v7.3');
  fprintf('done\n');
end

end


% ------------------------------------------------------------------------
function rec = attach_proposals(voc_rec, boxes, class_to_id)
% ------------------------------------------------------------------------

% change selective search order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
boxes = boxes(:, [2 1 4 3 5]);

%           gt: [2108x1 double]
%      overlap: [2108x20 single]
%      dataset: 'voc_2007_trainval'
%        boxes: [2108x4 single]
%         feat: [2108x9216 single]
%        class: [2108x1 uint8]
if isfield(voc_rec, 'objects')
  gt_boxes = cat(1, voc_rec.objects(:).bbox);
  all_boxes = cat(1, gt_boxes, boxes(:,1:4));
  gt_classes = class_to_id.values({voc_rec.objects(:).class});
  gt_classes = cat(1, gt_classes{:});
  num_gt_boxes = size(gt_boxes, 1);
  rec.scores = [ones(size(gt_boxes,1),1); boxes(:,5)];
else
  gt_boxes = [];
  all_boxes = boxes(:,1:4);
  gt_classes = [];
  num_gt_boxes = 0;
  rec.scores = boxes(:,5);
end
num_boxes = size(boxes, 1);

rec.gt = cat(1, true(num_gt_boxes, 1), false(num_boxes, 1));
rec.overlap = zeros(num_gt_boxes+num_boxes, class_to_id.Count, 'single');
for i = 1:num_gt_boxes
  rec.overlap(:, gt_classes(i)) = ...
      max(rec.overlap(:, gt_classes(i)), boxoverlap(all_boxes, gt_boxes(i, :)));
end
rec.boxes = single(all_boxes);
rec.weights = ones(size(rec.gt));
rec.feat = [];
rec.class = uint8(cat(1, gt_classes, zeros(num_boxes, 1)));
end

%%%%%%%%%%%%%%%%%
function rec = my_attach_proposals(pos_boxes, neg_boxes, mi, ma, experiment_dir)
    pos_boxes = pos_boxes(:, [2 1 4 3 5]);
    neg_boxes = neg_boxes(:,[2 1 4 3 5]);
    
%           gt: [2108x1 double]
%      overlap: [2108x20 single]
%      dataset: 'voc_2007_trainval'
%        boxes: [2108x4 single]
%         feat: [2108x9216 single]
%        class: [2108x1 uint8]

    num_pos_boxes = size(pos_boxes, 1);
    boxes = [pos_boxes;neg_boxes];
    num_boxes = size(boxes,1);

    rec.gt = cat(1, false(num_boxes, 1));
    rec.overlap = zeros(num_boxes, 1, 'single');
    rec.overlap(1:num_pos_boxes) =  1;
    rec.overlap(num_pos_boxes+1:end) = 0;
    rec.boxes = single(boxes);
%     rec.weights = abs(boxes(:,5)+0.2);
%     C = 0.8;
%     rec.weights = C*(boxes(:,5)-mi)/(ma-mi);
    rec.weights = calculate_weights(experiment_dir, boxes(:,5));
    rec.feat = [];
    rec.class = uint8(zeros(num_boxes, 1));
    
end

function rcnn_make_window_file(experiment_dir, imdb, out_dir)
% rcnn_make_window_file(imdb, out_dir)
%   Makes a window file that can be used by the caffe WindowDataLayer 
%   for finetuning.
%
%   The window file format contains repeated blocks of:
%
%     # image_index 
%     img_path
%     channels 
%     height 
%     width
%     num_windows
%     class_index overlap x1 y1 x2 y2
%     <... num_windows-1 more windows follow ...>

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

roidb = imdb.roidb_func(experiment_dir,imdb);

if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

% window_file = sprintf('%s/%s.txt', ...
%     out_dir, imdb.name);
% fid = fopen(window_file, 'wt');
window_file = sprintf('%s/%s.mat', ...
    out_dir, imdb.name);

channels = 3; % three channel images

res = cell(1,length(imdb.image_ids));

for i = 1:length(imdb.image_ids)
  tic_toc_print('make window file: %d/%d\n', i, length(imdb.image_ids));
  img_path = imdb.image_at(i);
  roi = roidb.rois(i);
  num_boxes = size(roi.boxes, 1);
  res{i}.image_id = imdb.image_ids{i};
  res{i}.num_boxes = num_boxes;
  res{i}.boxes = [];
%   fprintf(fid, '%s %d\n', imdb.image_ids{i}, num_boxes);
  for j = 1:num_boxes
     
    gt = roi.gt(j);
    [ov, label] = max(roi.overlap(j,:));
    % zero overlap => label = 0 (background)
    if ov < 1e-5        % 1e-5
      label = 0;
      ov = 0;
    end
    bbox = roi.boxes(j,:)-1;
    bbox = floor(bbox);
    if bbox(1)<0
        bbox(1) = 0;
    elseif bbox(1)>=imdb.sizes(i,2)
        bbox(1) = imdb.sizes(i,2)-1;
    end
    if bbox(2)<0
        bbox(2) = 0;
    elseif bbox(2) >= imdb.sizes(i,1)
            bbox(2) = imdb.sizes(i,1)-1;
    end
    if bbox(3)>=imdb.sizes(i,2)
        bbox(3) = imdb.sizes(i,2)-1;
    elseif bbox(3)<0
        bbox(3)=0;
    end
    if bbox(4)>=imdb.sizes(i,1)
        bbox(4) = imdb.sizes(i,1)-1;
    elseif bbox(4)<0
        bbox(4)=0;
    end
    
    res{i}.boxes = [res{i}.boxes; gt, ov, bbox(1), bbox(2), bbox(3), bbox(4)];
%     fprintf(fid, '%d %.3f %d %d %d %d\n', ...
%         gt, ov, bbox(1), bbox(2), bbox(3), bbox(4));
  end
end
save(window_file, 'res');
% fclose(fid);

function res = imdb_eval_common(experiment_dir, cls, boxes, imdb, suffix, nms_thresh)
% res = imdb_eval_voc(cls, boxes, imdb, suffix)
%   Use the VOCdevkit to evaluate detections specified in boxes
%   for class cls against the ground-truth boxes in the image
%   database imdb. Results files are saved with an optional
%   suffix.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% Add a random string ("salt") to the end of the results file name
% to prevent concurrent evaluations from clobbering each other
use_res_salt = true;
% Delete results files after computing APs
rm_res = false;
% comp4 because we use outside data (ILSVRC2012)
comp_id = 'comp4';

% save results
if ~exist('suffix', 'var') || isempty(suffix) || strcmp(suffix, '')
  suffix = '';
else
  if suffix(1) ~= '_'
    suffix = ['_' suffix];
  end
end

if ~exist('nms_thresh', 'var') || isempty(nms_thresh)
  nms_thresh = 0.3;
end

cache_dir = [experiment_dir 'results/'];
VOCopts  = imdb.details.VOCopts;
image_ids = imdb.image_ids;
test_set = VOCopts.testset;


addpath(fullfile(VOCopts.datadir, 'VOCcode')); 

if use_res_salt
  prev_rng = rng;
  rng shuffle;
  salt = sprintf('%d', randi(100000));
  res_id = [comp_id '-' salt];
  rng(prev_rng);
else
  res_id = comp_id;
end
res_fn = sprintf(VOCopts.detrespath, imdb.name, cls);
% res_fn = sprintf('experiments/fastfood/%s_%s.txt', res_id, cls);

% write out detections in PASCAL format and score
fid = fopen(res_fn, 'w');
for i = 1:length(image_ids);
  bbox = boxes{i};
  keep = nms(bbox, nms_thresh);
  bbox = bbox(keep,:);
  for j = 1:size(bbox,1)
        fprintf(fid, '%s %f %d %d %d %d\n', image_ids{i}, bbox(j,end), bbox(j,1:4));
%         fprintf(fid, '%d %d %d %d %d %f\n', i, bbox(j,1:4), bbox(j,end));
  end
  res_boxes{i} = bbox;   % add by wsf
end
fclose(fid);

save([experiment_dir 'results/res_boxes_' imdb.name '.mat'],'res_boxes');

recall = [];
prec = [];
ap = 0;
ap_auc = 0;

do_eval = true;
if do_eval
  % Bug in VOCevaldet requires that tic has been called first
  tic;
  [recall, prec, ap] = VOCevaldet(VOCopts, res_id, cls, true);
  

  ap_auc = xVOCap(recall, prec);

%   force plot limits
  ylim([0 1]);
  xlim([0 1]);

  print(gcf, '-djpeg', '-r0', ...
      [cache_dir cls '_pr_' imdb.name suffix '.jpg']);
end
fprintf('!!! %s : %.4f %.4f\n', cls, ap, ap_auc);

res.recall = recall;
res.prec = prec;
res.ap = ap;
res.ap_auc = ap_auc;

save([cache_dir cls '_pr_' imdb.name suffix], ...
    'res', 'recall', 'prec', 'ap', 'ap_auc');

% DET curve
% [EER confInterEER OP confInterOP] = VOCevaldet_DET(VOCopts, res_id, cls, true);


if rm_res
  delete(res_fn);
end


rmpath(fullfile(VOCopts.datadir, 'VOCcode')); 

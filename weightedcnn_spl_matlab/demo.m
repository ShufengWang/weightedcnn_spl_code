%% initialize
clear all,clc;

experiment_name = 'mit_weightedcnn_spl';
experiment_dir = ['fastfood_experiments/' experiment_name '/'];


addpath(experiment_dir);
addpath([experiment_dir 'VOCcode/']);
addpath([experiment_dir 'extract_feature/']);
addpath([experiment_dir 'imdb/']);


nms_thres = 0.3;
max_per_image =2000;

%% preprocessing
resize_factor = 4;
% resize_image;
% 
createxml;
parseCandidates;

%% visualize gt bounding box
subset = 'unlabel';

img_dir = [experiment_dir 'dataset/images/'];
anno_dir = [ experiment_dir 'dataset/annotations/'];
images = textread([experiment_dir 'dataset/imagesets/' subset '.txt'], '%s');


for i = 1:length(images)
    img = imread([img_dir images{i} '.png']);
    gt = VOCreadrecxml([anno_dir subset '/' images{i} '.xml']);
    num = length(gt.objects);
    boxes = [];
    for j = 1:num
        boxes(j,:) = gt.objects(j).bbox;
    end
    showboxes(img,boxes);
    
    pause;
end

%% visualize candidates bounding box
subset = 'test';

img_dir = [ experiment_dir 'dataset/images/'];

img_ids = textread([experiment_dir 'dataset/imagesets/' subset '.txt'], '%s');
load([experiment_dir 'data/candidates/' subset '.mat']);

for i = 1:length(img_ids)
    img = imread([img_dir img_ids{i} '.png']);
    box = boxes{i};
    showboxes(img,box(1:10,[2 1 4 3]));
    pause;
end


%% create window file

subsets = {'train', 'val', 'unlabel', 'test'};

for id=1:length(subsets)
    
    subset = subsets{id};

    imdb = imdb_from_common(experiment_dir, subset);
 
    win_file_outdir = [experiment_dir 'window_files'];
    rcnn_make_window_file(experiment_dir,imdb, win_file_outdir);
    
end

% merge train and unlabel subset

%% forward cnn


reset(gpuDevice(4));

net_file     = [experiment_dir 'data/caffe_nets/finetune_trainval_iter_2000.caffemodel'];
cache_name   = ['v1_finetune_iter_2000'];
crop_mode    = 'warp';
crop_padding = 16;
 

% ------------------------------------------------
% subsets = {'train', 'test', 'val', 'unlabel'};
% subsets = {'train', 'val', 'test'};
subsets = {'test'};

for id=1:length(subsets)
    
    subset = subsets{id};
    imdb = imdb_from_common(experiment_dir, subset);

    rcnn_exp_forward_common(imdb,...
            experiment_dir, ...
            'crop_mode', crop_mode, ...
            'crop_padding', crop_padding, ...
            'net_file', net_file, ...
            'cache_name', cache_name);
    
end
reset(gpuDevice(4));


%% calculate accuracy
subset = 'test';

imdb_test = imdb_from_common(experiment_dir,subset);

load([experiment_dir 'results/rcnn_model.mat']);

% conf = rcnn_config('sub_dir', imdb_test.name);
cache_dir = [experiment_dir 'results/'];

image_ids = imdb_test.image_ids;

feat_opts = rcnn_model.training_opts;
num_classes = length(rcnn_model.classes);

try
  aboxes = cell(num_classes, 1);
  for i = 1:num_classes
    load([cache_dir rcnn_model.classes{i} '_boxes_' imdb_test.name ]);
    aboxes{i} = boxes;
  end
catch
  aboxes = cell(num_classes, 1);
  box_inds = cell(num_classes, 1);
  for i = 1:num_classes
    aboxes{i} = cell(length(image_ids), 1);
    box_inds{i} = cell(length(image_ids), 1);
  end

  % heuristic that yields at most 100k pre-NMS boxes
  % per 2500 images
  % max_per_set = ceil(100000/2500)*length(image_ids);
  % max_per_image = 100;
  max_per_set = max_per_image*length(image_ids);
  
  
  top_scores = cell(num_classes, 1);
  thresh = -inf(num_classes, 1);
  box_counts = zeros(num_classes, 1);

  if ~isfield(rcnn_model, 'folds')
    folds{1} = 1:length(image_ids);
  else
    folds = rcnn_model.folds;
  end

  count = 0;
  for f = 1:length(folds)
    for i = folds{f}
      count = count + 1;
      fprintf('%s: test (%s) %d/%d\n', procid(), imdb_test.name, count, length(image_ids));
      d = rcnn_load_cached_softmax_features(experiment_dir, feat_opts.cache_name, ...
          imdb_test.name, image_ids{i});
      if isempty(d.feat)
        continue;
      end
      
      zs = d.feat;
      

      for j = 1:num_classes
        boxes = d.boxes;
%    here is j+1 not j, because class 1 means background     
        z = zs(:,j+1);
        I = find(~d.gt & z > thresh(j));
        boxes = boxes(I,:);
        scores = z(I);
        aboxes{j}{i} = cat(2, single(boxes), single(scores));
        [~, ord] = sort(scores, 'descend');
        ord = ord(1:min(length(ord), max_per_image));
        aboxes{j}{i} = aboxes{j}{i}(ord, :);
        box_inds{j}{i} = I(ord);

        box_counts(j) = box_counts(j) + length(ord);
        top_scores{j} = cat(1, top_scores{j}, scores(ord));
        top_scores{j} = sort(top_scores{j}, 'descend');
        if box_counts(j) > max_per_set
          top_scores{j}(max_per_set+1:end) = [];
          thresh(j) = top_scores{j}(end);
        end
      end
    end
  end

  for i = 1:num_classes
    % go back through and prune out detections below the found threshold
    for j = 1:length(image_ids)
      if ~isempty(aboxes{i}{j})
        I = find(aboxes{i}{j}(:,end) < thresh(i));
        aboxes{i}{j}(I,:) = [];
        box_inds{i}{j}(I,:) = [];
      end
    end

    save_file = [cache_dir rcnn_model.classes{i} '_boxes_' imdb_test.name];
    boxes = aboxes{i};
    inds = box_inds{i};
    save(save_file, 'boxes', 'inds');
    clear boxes inds;
  end
end

% % ------------------------------------------------------------------------
% % Peform AP evaluation
% % ------------------------------------------------------------------------
for model_ind = 1:num_classes
  cls = rcnn_model.classes{model_ind};
  res(model_ind) = imdb_test.eval_func(experiment_dir, cls, aboxes{model_ind}, imdb_test,'', nms_thres);
end

fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Results:\n');
aps = [res(:).ap]';
disp(aps);
disp(mean(aps));
fprintf('~~~~~~~~~~~~~~~~~~~~\n');


%% visualize result bounding box
subset = 'test';
img_dir = [experiment_dir 'dataset/images/'];

img_ids = textread([experiment_dir 'dataset/imagesets/' subset '.txt'], '%s');
load( [experiment_dir 'results/res_boxes_test.mat']);
load( [experiment_dir 'results/thres_fppi1.mat']);

for i = 1:length(img_ids)
    img = imread([img_dir img_ids{i} '.png']);
    
    img_height = size(img,1);
    img_width = size(img,2);

    top = floor(0.2 * img_height);
    left = floor(0.2 * img_width);
    
    box = res_boxes{i};
    idx = find(box(:,end)<thres_fppi1);
    box(idx,:) = [];
    box(:,[1,3]) = box(:,[1,3]) - left;
    box(:,[2,4]) = box(:,[2,4]) - top;
    
    w = box(:,3)-box(:,1);
    h = box(:,4)-box(:,2);
    box(:,1) = box(:,1) + w*12/60;
    box(:,3) = box(:,3) - w*12/60;
    box(:,2) = box(:,2) + h*12/120;
    box(:,4) = box(:,4) - h*12/120;
    img = img(top+1:img_height-top, left+1:img_width-left,:);
    disp(i)
    showboxes(img,box,[experiment_dir 'results/detection_results/', img_ids{i} '.png']);
    img = imread([experiment_dir 'results/detection_results/', img_ids{i} '.png']);
    img = img(2:end-1, 2:end-1,:);
    imwrite(img, [experiment_dir 'results/detection_results/', img_ids{i} '.png']);
end




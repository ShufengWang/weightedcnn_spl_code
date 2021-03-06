function rcnn_exp_forward_common(imdb,experiment_dir, varargin)

ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addOptional('start', 1, @isscalar);
ip.addOptional('end', 0, @isscalar);
ip.addOptional('crop_mode', 'warp', @isstr);
ip.addOptional('crop_padding', 16, @isscalar);
ip.addOptional('net_file', ...
    [experiment_dir 'data/caffe_nets/finetune_trainval_iter_70k'], ...
    @isstr);
ip.addOptional('cache_name', ...
    'v1_finetune_trainval_iter_70000', @isstr);

ip.parse(imdb, varargin{:});
opts = ip.Results;

opts.net_def_file = './model-defs/rcnn_output_softmax.prototxt';

image_ids = imdb.image_ids;
if opts.end == 0
  opts.end = length(image_ids);
end

% Where to save feature cache
opts.output_dir = [experiment_dir 'feat_cache/' opts.cache_name '/' imdb.name '/'];

mkdir_if_missing(opts.output_dir);

% Log feature extraction
timestamp = datestr(datevec(now()), 'dd.mmm.yyyy:HH.MM.SS');
diary_file = [opts.output_dir 'rcnn_cache_softmax_features_' timestamp '.txt'];
diary(diary_file);
fprintf('Logging output in %s\n', diary_file);

fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Feature caching options:\n');
disp(opts);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

% load the region of interest database
roidb = imdb.roidb_func(experiment_dir,imdb);

rcnn_model = rcnn_create_model(opts.net_def_file, opts.net_file);
rcnn_model = rcnn_load_model(rcnn_model);
rcnn_model.detectors.crop_mode = opts.crop_mode;
rcnn_model.detectors.crop_padding = opts.crop_padding;

total_time = 0;
count = 0;
for i = opts.start:opts.end
  fprintf('%s: cache features: %d/%d\n', procid(), i, opts.end);

  save_file = [opts.output_dir image_ids{i} '.mat'];
  if exist(save_file, 'file') ~= 0
    fprintf(' [already exists]\n');
    continue;
  end
  count = count + 1;

  tot_th = tic;

  d = roidb.rois(i);
  im = imread(imdb.image_at(i));

  th = tic;
  d.feat = rcnn_features(im, d.boxes, rcnn_model);
  fprintf(' [features: %.3fs]\n', toc(th));

  th = tic;
  save(save_file, '-struct', 'd');
  fprintf(' [saving:   %.3fs]\n', toc(th));

  total_time = total_time + toc(tot_th);
  fprintf(' [avg time: %.3fs (total: %.3fs)]\n', ...
      total_time/count, total_time);
end
rcnn_model.training_opts = opts;
rcnn_model.classes = imdb.classes;
save([experiment_dir 'results/rcnn_model.mat'], 'rcnn_model');

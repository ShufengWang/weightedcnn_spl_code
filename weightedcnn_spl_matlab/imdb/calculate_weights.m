function [ weights ] = calculate_weights( experiment_dir, scores )
%CALCULATE_WEIGHTS Summary of this function goes here
%   Detailed explanation goes here
    imdb = imdb_from_common(experiment_dir, 'train');
    roidb = imdb.roidb_func(experiment_dir,imdb);
    
    s = [];
    weights = [];
    for i = 1:length(imdb.image_ids)
        img_path = imdb.image_at(i);
        roi = roidb.rois(i);
        s = [s;roi.scores];
    end
    d = 0;
    for i = 1: length(scores)
        d = max(d,max(1./(abs(s-scores(i))+0.0001)));
    end
    C = 0.8;
    for i = 1: length(scores)
        weights(i) = C*max(1./(abs(s-scores(i))+0.0001))/d;
    end
    
end


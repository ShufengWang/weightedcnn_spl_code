%%

num_candidates = 600;
% num_candidates = inf;


% for id=1:length(subsets)
%     subset = subsets{id};
%     if ~exist([experiment_dir 'dataset/annotations/' subset],'dir')
%         mkdir([experiment_dir 'dataset/annotations/' subset])
%         addpath([experiment_dir  'dataset/annotations/' subset]);
%     end
% end

dataset_dir = [experiment_dir 'dataset/'];

addpath([dataset_dir 'candidates_txt/']);
addpath([dataset_dir 'imagesets/']);

%%
subsets = {'train', 'test', 'val', 'unlabel'};

total_candidates = 0;
total_image = 0;
ma = 0;
mi = inf;

for id=1:length(subsets)
    subset = subsets{id};
    
    image_list = textread([dataset_dir 'imagesets/' subset '.txt'],'%s');
    num_image = length(image_list);

    boxes = {};
    images = {};
    pos_boxes = {};
    neg_boxes = {};
 
    for i = 1:num_image
        images{1,i} = [image_list{i} '.png'];
        [x1,y1,x2,y2,score] = textread([dataset_dir 'candidates_txt/' image_list{i} '.txt'],...
            '%d%d%d%d%f');
%         x2 = x1 + w;
%         y2 = y1 + h;
        img = imread([dataset_dir 'resized_images/' image_list{i} '.png']);
        imgh = size(img,1);
        imgw = size(img,2);
        x1 = x1-0.2*imgw;
        x2 = x2-0.2*imgw;
        y1 = y1-0.2*imgh;
        y2 = y2-0.2*imgh;
        boxes{1,i} = [y1,x1,y2,x2,score];
        boxes{1,i}(:,1:4) = int32(boxes{1,i}(:,1:4)/resize_factor);
        
        [~,I] = sort(-score);
        boxes{1,i} = boxes{1,i}(I,:);
        score = score(I);
        
        if(strcmp(subset, 'unlabel'))
            boxes{1,i} = boxes{1,i}(1:min(3*num_candidates, end),:);
            score = score(1:min(3*num_candidates, end));
        else
            boxes{1,i} = boxes{1,i}(1:min(num_candidates, end),:);
            score = score(1:min(num_candidates, end));
        end
        
        if size(score,1)>0
            ma = max(ma,score(1));
            mi = min(mi,score(end));
        end
        
        total_candidates = total_candidates + size(boxes{i},1);
        total_image  = total_image + 1;
        
    end
    save([experiment_dir 'data/candidates/' subset '.mat'],'boxes','images');

end

save([experiment_dir 'data/candidates/mi_ma_score.mat'],'mi','ma');
fprintf('total candidates: %d\n  total images:  %d\n average candidates per image: %d\n',...
        total_candidates, total_image, floor(total_candidates/total_image));



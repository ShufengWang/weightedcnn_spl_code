
%%
subsets = {'train', 'test', 'val','unlabel'};  % not include 'unlabel' subset


for id=1:length(subsets)
    
    subset = subsets{id};

    if ~exist([experiment_dir 'dataset/annotations/' subset],'dir')
        mkdir([experiment_dir 'dataset/annotations/' subset])
        addpath([experiment_dir  'dataset/annotations/' subset]);
    end
end
if ~exist([experiment_dir  'dataset/imagesets/'],'dir')
    mkdir([experiment_dir  'dataset/imagesets/'])
    addpath([experiment_dir  'dataset/imagesets/']);
end

%% create xml for subset

for id=1:length(subsets)
    
    subset = subsets{id};
    image_list = textread([experiment_dir 'dataset/imagesets/' subset '_list.txt'],'%s');

    num_image = length(image_list);

    fid = fopen([experiment_dir 'dataset/imagesets/' subset '.txt'],'w');
    for i = 1:num_image
        [img_id, ~] = strtok(image_list{i},'.');
        num = str2num(img_id);
        fprintf(fid,'%s\n',img_id);

    end
    fclose(fid);

    image_list = textread([experiment_dir 'dataset/imagesets/' subset '.txt'],'%s');
    num_image = length(image_list);


    for i = 1:num_image

        im_name = [experiment_dir 'dataset/images/' image_list{i} '.png'];
        im = imread(im_name);
        rec = struct;
        rec.folder = experiment_dir;
        rec.filename = [image_list{i} '.png'];
        rec.source.database = experiment_dir;
        rec.source.annotation = experiment_dir;
        rec.source.image = 'wsf';
        rec.source.wsfid = '1';
        rec.own.wsfid = 'ok';
        rec.own.name = 'wang';
        rec.size.width = size(im,2);
        rec.size.height = size(im,1);
        rec.size.depth = size(im,3);
        rec.segmented = '0';

        [x1,y1,x2,y2] = textread([experiment_dir 'dataset/annotations_txt/'  image_list{i} '.txt'],'%d%d%d%d');
        
        
        num_bb = size(x1,1);
        rec.object = struct;

        for j=1:num_bb
                rec.object(j).name = 'Pedestrian';
                rec.object(j).pose = 'Left';
                rec.object(j).truncated = '1';
                rec.object(j).difficult = '0';
                rec.object(j).bndbox.xmin = x1(j);
                rec.object(j).bndbox.ymin = y1(j);
                rec.object(j).bndbox.xmax = x2(j);
                rec.object(j).bndbox.ymax = y2(j);


        end
        stru = struct;
        stru.annotation = rec;
        VOCwritexml(stru, sprintf([experiment_dir 'dataset/annotations/' subset '/%s.xml'],image_list{i}));
        if mod(i,100)==0 
            disp(i);
        end
    end

end
        


        
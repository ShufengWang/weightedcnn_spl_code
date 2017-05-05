%% images
image_dir = [experiment_dir 'dataset/resized_images/'];
list = dir([image_dir '*.png']);

img_out_dir = [experiment_dir 'dataset/images/'];
if ~exist(img_out_dir,'dir')
    mkdir(img_out_dir);
end

for i = 1:length(list)
   im = imread([image_dir list(i).name]);
   sz = size(im);
   sz = int32(sz/resize_factor);
   new_im = imresize(im, [sz(1),sz(2)]);
   imwrite(new_im, [img_out_dir list(i).name]);
end

%% annotations
fid = fopen([image_dir 'annotation.txt'],'r');
out_dir = [experiment_dir 'dataset/annotations_txt/'];

while true
    s = fgetl(fid);
    if(~ischar(s)) break;
    end
    [img_name s] = strtok(s, ' ');
%     fout = fopen([out_dir strtok(img_name, '.') '.txt'],'w');
    img = imread([img_out_dir img_name]);
    boxes = [];
    while(length(s)>0)
        s = s(2:end);
        if(length(s)==0) break; end
        [a, s] = strtok(s,' ');
        [x1, a] = strtok(a,':');
        a = a(2:end);
        [y1, a] = strtok(a,':');
        a = a(2:end);
        [x2,a] = strtok(a,':');
        y2 = a(2:end);
        
        x1 = str2num(x1);
        y1 = str2num(y1);
        x2 = str2num(x2);
        y2 = str2num(y2);
        h = y2-y1;
        w = x2-x1;
        x1 = (x1+0.5*w-0.333*h);
        y1 = (y1-0.167*h);
        x2 = (x1+0.667*h);
        y2 = (y1+1.334*h);
        
        x1 = int32(x1/resize_factor);
        y1 = int32(y1/resize_factor);
        x2 = int32(x2/resize_factor);        
        y2 = int32(y2/resize_factor);
        if(x1<0) x1 = 0; end
        if(y1<0) y1 = 0; end
        if(x2>size(img,2)) x2 = size(img, 2); end
        if(y2>size(img,1)) y2 = size(img, 1); end
   
        boxes = [boxes;x1,y1,x2,y2];
    end
    dlmwrite([out_dir strtok(img_name, '.') '.txt'], boxes, ' ');
end



experiment_name = 'mit_new_more_unlabel';
experiment_dir = ['fastfood_experiments/' experiment_name '/'];
cache_name   = ['v1_finetune_iter_2000'];

subsets = {'test'};
for id=1:length(subsets)
    
    subset = subsets{id};
    imdb = imdb_from_common(experiment_dir, subset);
   
    image_ids = imdb.image_ids;
    
    output_dir = [experiment_dir 'feat_cache/' cache_name '/' subset '_txt/'];
    mkdir_if_missing(output_dir);
    count = 0;
    for i = 1:length(image_ids)
        fp = fopen([output_dir image_ids{i} '.txt'],'w');
        count = count + 1;
        fprintf('mat2txt: %s %d/%d\n', imdb.name, count, length(image_ids));
        d = rcnn_load_cached_softmax_features(experiment_dir, cache_name, ...
            imdb.name, image_ids{i});
        if isempty(d.feat)
            fclose(fp);
            continue;
        end
        zs = d.feat;
        z = zs(:,2);
        I = find(~d.gt);
        z = z(I);
        boxes = d.boxes(I,:);
        
        top = int32(imdb.sizes(i,1)/7);
        left = int32(imdb.sizes(i,2)/7);
        
        for j = 1:length(z)
            x1 = boxes(j,1)-left;
            y1 = boxes(j,2)-top;
            w = boxes(j,3)-boxes(j,1);
            h = boxes(j,4)-boxes(j,2);
            fprintf(fp, '%d %d %d %d %f\n', x1, y1, w, h, z(j));
            
        end
        fclose(fp);
        
    end
end
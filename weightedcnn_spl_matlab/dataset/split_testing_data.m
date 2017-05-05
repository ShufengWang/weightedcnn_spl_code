image_list = textread('unlabel_list.txt','%s');
fp = fopen('fastfood_experiments/mit_new_more_unlabel/dataset/imagesets/unlabel_list.txt','w');
for i = 1:length(image_list)
    if(mod(i,5)==1 || mod(i,5)==0)
        fprintf(fp, '%s\n',image_list{i});
    end
end
fclose(fp);
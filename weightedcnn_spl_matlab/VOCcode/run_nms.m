function [ boxes ] = run_nms( boxes, nms_thres )
%RUN_NMS Summary of this function goes here
%   Detailed explanation goes here
    num_boxes = size(boxes,1);
    select = ones(num_boxes,1);
    for i = 1:num_boxes
        if(select(i)==1)
            for j = i+1:num_boxes
                if(iou(boxes(i,:),boxes(j,:))>nms_thres)
                    select(j) = 0;
                end
            end
        end
    end
    boxes(find(select==0),:)=[];
end


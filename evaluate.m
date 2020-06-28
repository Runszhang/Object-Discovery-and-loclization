clear all


%%%%%%%%%%%%%%%%%%%%%%%%%%voc2007%%%%%%%%%%%%%%%%%%%%%%%%%%corlocc%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 Smap_path=[Your support_map path 'S_map/'];
 annotation_path=[Your VOC  PATH '/VOC2007/Annotations/'];
 voc_path=[Your VOC '6x2' PATH '/VOC2007_6x2/'];
 voc=dir(voc_path);



corloc=cell(13,2);
for i=3:length(voc)
    
    sub_class=voc(i).name;
    sub_class_path=fullfile(voc_path,sub_class); 
    voc_set=dir(fullfile(sub_class_path,'/','*.jpg'));
    pre_true_x=0;
    for img_j=1:length(voc_set)
        img_name=voc_set(img_j).name;
        
        im=imread(fullfile(sub_class_path,img_name));
        [h,w,~]=size(im);
        cutnum=strfind(img_name,'.');
        imagename=img_name(1:cutnum-1);
        ann_path=[annotation_path '/', imagename '.xml'];         
        anno = PASreadrecord(ann_path);
        
        obj_class_list={anno.objects.class};
        class_num=numel(unique(obj_class_list));%%class number C
        
        load(fullfile(sub_class_path,[imagename '.mat']));%%load gt
        load(fullfile(Smap_path, [imagename '.mat']));%%load support map
        
        
       % imshow(im);
        gt_boxes = cell2mat(bbox_list');
        gt_boxes(:, 3:4) = gt_boxes(:, 3:4) - gt_boxes(:, 1:2) + 1;
        %rectangle('position',gt_boxes(1,:),'EdgeColor','r');
        Igray=imresize(S_map,[h,w],'nearest');
        Igray(Igray>0)=1;
        Igray(Igray<=0)=0;
     %figure(2)
        cc = bwconncomp(Igray); 
        numPixel = cellfun(@numel,cc.PixelIdxList);
        [aa,bb] =sort(numPixel);
        pre_rect=[];
           

        for e=1:length(numPixel)
            highlight =zeros(h,w);
            conn_idx1 = bb(e);
            highlight(cc.PixelIdxList{conn_idx1}) = 1;
            img_reg=regionprops(highlight,'area', 'boundingbox');
            pre_rect(e,:)=cat(1, img_reg.BoundingBox);
        end
        
        pre_rect(all(pre_rect==0,2),:)=[];
        
        pre_x=size(pre_rect,1);%%the number of predict bbox
        if pre_x>class_num
            pre_rect=pre_rect((end-(class_num-1)):1:end,:);
        end
        
        
%         if pre_x>1
%             pre_rect=pre_rect(end-1:1:end,:);
%         end
         
         gt_rects=gt_boxes;
         for ii=1:size(pre_rect,1)
             Iou=[];
             for t=1:size(gt_rects,1)
                 Iou(t)= IoU(pre_rect(ii,:), gt_rects(t,:));
             end
             if max(Iou)>=0.5
                pre_true_x=pre_true_x+1;
               break 
             end  
             
         end
   
    end
    corloc_cls=pre_true_x/length(voc_set);
    corloc{i-2,1}=sub_class;
    corloc{i-2,2}=corloc_cls;
    
   % ex_time(i,1) = toc;
disp(['Extracing ' ': ' num2str(i-2)  sub_class 'used '  's ...']);   
end
corloc_mean = sum(cell2mat(corloc(:,2)))/12;
corloc{13,2} = corloc_mean;
corloc{13,1} = 'mean';



% The codes are written by Runsheng Zhang. For any problem concerning the code, please feel free to contact Mr. Zhang.
% This packages are free for academic usage. You can run them at your own risk. For other purposes, please contact Prof. Yaping Huang.

% The codes are corresponding to the *mined patterns* procedure
% of the proposed  method.
clear all

addpath([Your Model_PATH  'Model']);
addpath([Your Matconvnet_PATH '/matconvnet-1.0-beta23']);
% Setting the GPU device
gpu = 3;
g = gpuDevice(gpu);
reset(g);




% The pre-trained model--VGG-16
opt.model = 'imagenet-vgg-verydeep-16';

opt.thr = 'mean';

% load CNN model
run([Your Matconvnet_PATH '/matconvnet-1.0-beta23/matlab/vl_setupnn.m']);
net = load(['/Model/' opt.model '.mat']);
net.layers(end-5:end)=[]; % Removing the fully connected layers
net = vl_simplenn_move(net, 'gpu') ;
disp('CNN model is ready ...');




%%%%%%VOC2007 6X2%%%%%%%%%%%%

 dirs=dir([VOC2007_DATASET_PATH '/VOC2007/ImageSets/Main/*_trainval.txt']);
 voc_path=[Your VOC '6x2' PATHT '/VOC2007_6x2/'];
 voc=dir(voc_path);
 path=[Your genetate data path '/generate_data/'];



% Using the RGB average values obtained from ImageNet

net.normalization.averageImage = ones(224,224,3);
net.normalization.averageImage(:,:,1) = net.normalization.averageImage(:,:,1) .* net.meta.normalization.averageImage(1,1);
net.normalization.averageImage(:,:,2) = net.normalization.averageImage(:,:,2) .* net.meta.normalization.averageImage(1,2);
net.normalization.averageImage(:,:,3) = net.normalization.averageImage(:,:,3) .* net.meta.normalization.averageImage(1,3);
imdb.averageImage = net.normalization.averageImage;

ex_time = [];

for i=3:length(voc)
    sub_class=voc(i).name;
    sub_class_path=fullfile(voc_path,sub_class); 
    voc_set=dir(fullfile(sub_class_path,'/','*.jpg'));
    
    for img_j=1:length(voc_set)
        img_name=voc_set(img_j).name;
        
        im=imread(fullfile(sub_class_path,img_name));

        cutnum=strfind(img_name,'.');
        imagename=img_name(1:cutnum-1);
        im_ = single(im);
        
        [h,w,~] = size(im_);
        
        if min(h,w) > 700
        im_ = imresize(im_, [h*(700/min(h,w)) w*(700/min(h,w))]);
        end
        [h,w,c] = size(im_);
        if  c > 2
            im_ = im_ - imresize(imdb.averageImage,[h,w]) ;
        else    
            im_ = bsxfun(@minus,im_,imresize(imdb.averageImage,[h,w])) ;
        end
        %%%%%%%%%%Extracting cnn feature map%%%%%%%%%%%
        res = vl_simplenn(net, gpuArray(im_)) ;
        tmp_1 = gather(res(32).x);%net pool5
        tmp_2 = gather(res(29).x);%net relu 5_2
        if ~exist(fullfile(path,'/pool5_feature'));
            mkdir(fullfile(path,'/pool5_feature'));
        end
        save(fullfile(path ,'/pool5_feature',[imagename,'.mat']),'tmp_1','-v7.3');
        if ~exist(fullfile(path,'/relu5_feature'));
           mkdir(fullfile(path,'/relu5_feature'));
        end
        save(fullfile(path ,'/relu5_feature',[imagename,'.mat']),'tmp_2','-v7.3');
       %%%%%%Mining patterns%%%%%%%%%%%%%%%%%%%%%%
        tmp_featmap = tmp_1;
        Hrelu=size(tmp_2,1);
        Wrelu=size(tmp_2,2);
        Re_tmp_1=imresize(tmp_1, [Hrelu, Wrelu]);
        tmp_3=cat(3,Re_tmp_1,tmp_2);%%pool5+relu5
        tmp_1=tmp_3;%pool5+relu5
        Apool=zeros(size(tmp_1,1),size(tmp_1,2));
        Acombined=[];
        Apoolcombined=[];
        tmp_featmap = tmp_1;
        for j=1:size(tmp_1,3)
            Apool=tmp_1(:,:,j);
            Bpool=Apool(:);
            Apoolcombined=[Apoolcombined,Bpool];
        end
  
        [feaSorted,feaIndex] = sort(Apoolcombined,'descend');
        file = false(size(Apoolcombined));
        for ii = 1:size(Apoolcombined,2)
            Trancow=Apoolcombined(:,ii);
            Trancow=Trancow(find(Trancow>0));
            mean_Tran=mean(Trancow);
            numTopActivation=sum(Trancow>mean_Tran);    
            file(feaIndex(1:numTopActivation,ii),ii)=1;
        end
 
        if ~exist(fullfile(path,'/invertFile'));
            mkdir(fullfile(path,'/invertFile'));
        end
        save(fullfile(path ,'/invertFile',[imagename,'.mat']),'file','-v7.3');

   %%%%%%%%%%%%%%transcation creation%%%%%%%%%%%
        if ~exist(fullfile(path,'/transFile'));
           mkdir(fullfile(path,'/transFile'));
        end  
        fileName = fullfile(path,'/transFile/',[imagename,'.txt']);
        fid = fopen(fileName,'w');
        for k = 1:size(file,2)
            v = find(file(:,k));
            
            if length(v)~=0
               for j = 1:length(v)
                   if j==length(v)
                      fprintf(fid,'%d',v(j));   
                   else   
                      fprintf(fid,'%d',v(j));
                      fprintf(fid,',');
                   end
               end       
              fprintf(fid,'\n');
            else 
              continue
            end
        end
        
        fclose(fid);
   
        if ~exist(fullfile(path,'/rule'));
           mkdir(fullfile(path,'/rule'));
        end
        if ~exist(fullfile(path,'/multi_rule'));
          mkdir(fullfile(path,'/multi_rule'));
        end  
       
       
        supp=[7 8 9];%%support value
        for sup_i=1:length(supp)  
            sup=supp(sup_i);
            inputFile = fullfile(path,'/transFile',[imagename,'.txt']);
            outputFile = fullfile(path,'/rule',[imagename,'.txt']);
            outputFile_multi = fullfile(path,'/multi_rule',[imagename,'.txt']);
            options = ['./apriori -ts -s',num2str(sup),' -m',num2str(1),' -n',num2str(1)];
            system([options,' ',inputFile,' ',outputFile]);
            options = ['./apriori -ts -s',num2str(sup),' -m',num2str(2),' -n',num2str(2)];
            system([options,' ',inputFile,' ',outputFile_multi]);
%%%%%%%%%%%%%%%%%%%%%%%%%the frequency of mined region %%%%%%%%%%%%%%%%%%%%%%%  
            fid=fopen(fullfile(path,'/multi_rule',[imagename,'.txt']),'r');
            C1 = textscan(fid,'%s','delimiter','\n');
            Nrow = length(C1{1});
            rule1=cell(Nrow,1);
            cluster1 = cell(Nrow,1);
            clusterString1 = cell(Nrow,1);
           fclose(fid);
            fid=fopen(fullfile(path,'/multi_rule',[imagename,'.txt']),'r');
 
            for iii=1:Nrow
                line=fgetl(fid);
                en=strfind(line,'(');    
                substr1=line(1:en-2);      
                rule1{iii}=str2num(substr1);
            end
  
            index1 = false(1,size(Apoolcombined,1));
            for inum=1:Nrow
                for iinum=1:length(rule1{inum});
                    index1(rule1{inum}(iinum))=1;
                end
            end           
           fclose(fid);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           fid = fopen(fullfile(path,'/rule',[imagename,'.txt']),'r');  
           C = textscan(fid,'%s','delimiter','\n');
           numRow = length(C{1});
           rule = cell(numRow,2);
           cluster = cell(numRow,1);
           clusterString = cell(numRow,1);
           fclose(fid);
           fid = fopen(fullfile(path,'/rule',[imagename,'.txt']),'r');  
           for iii = 1:numRow
               line = fgetl(fid);
               en=strfind(line,'(');
               ss=strfind(line,')');
               st=strfind(line,' ');
               substr=line(1:st-1);
               sub_sup=line(en+1:ss-1);
               rule{iii,1} = str2num(substr);
               rule{iii,2} = str2num(sub_sup);
           end
           fclose(fid);   
           index = false(1,size(Apoolcombined,1));
           index=double(index);
           for inum = 1:numRow
               index(rule{inum,1})=rule{inum,2};
        %
           end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           for in=1:length(index)
               if (index1(in)&index(in))==0
                  index(in)=0;
               end
           end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           Min_region=double(index)';
           S_map=reshape(Min_region,size(tmp_1,1),size(tmp_1,2)); %%support_map
           if ~exist(fullfile(path,'/support_mask/',num2str(sup)));
              mkdir(fullfile(path,'/support_mask/',num2str(sup)));
           end  
           save( fullfile(path,'/support_mask/',num2str(sup), [imagename '.mat']),'S_map','-v7.3')

           
        end
        disp(['Mined : ' num2str(img_j) 'th image (' num2str(j*100/length(voc)) '%) used '])
    end
    disp(['Mined: ' sub_class 'class'])
end
disp([' The dataset  is finished ...']);

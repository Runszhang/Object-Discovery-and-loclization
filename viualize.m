clear all         

%% Show the fig1 in the paper


im=imread(['example/','005783.jpg']);

im_ = single(im);
        
[h,w,~] = size(im_);
load(['example/005783.mat']);%mined support map

S_map = imresize(double(S_map),[h w]);  
curHeatMap=S_map;
curHeatMap = map2jpg(curHeatMap,[], 'jet');
           
curHeatMap = im2double(im)*0.5+curHeatMap*0.5;
imshow((curHeatMap));
ss2=im2uint8(mat2gray(curHeatMap));
imwrite(ss2,['example/001556.bmp']); 
clear VOCopts

% dataset
%
% Note for experienced users: the VOC2008-10 test sets are subsets
% of the VOC2010 test set. You don't need to do anything special
% to submit results for VOC2008-10.

VOCopts.dataset='MIT_Traffic';

% get devkit directory with forward slashes
devkitroot=strrep(fileparts(fileparts(mfilename('fullpath'))),'\','/');

% change this path to point to your copy of the PASCAL VOC data
VOCopts.datadir=[devkitroot '/'];

% change this path to a writable directory for your results
VOCopts.resdir=[devkitroot '/results/'];

% change this path to a writable local directory for the example code
VOCopts.localdir=[devkitroot '/local/'];

% initialize the training set

VOCopts.trainset='train'; % use train for development
% VOCopts.trainset='trainval'; % use train+val for final challenge

% initialize the test set

VOCopts.testset='val'; % use validation data for development test set
% VOCopts.testset='test'; % use test set for final challenge

% initialize main challenge paths

VOCopts.annopath=[VOCopts.datadir 'dataset/annotations/%s/%s.xml'];
VOCopts.imgpath=[VOCopts.datadir 'dataset/images/%s.png'];
VOCopts.imgsetpath=[VOCopts.datadir 'dataset/imagesets/%s.txt'];
VOCopts.clsimgsetpath=[VOCopts.datadir 'dataset/imagesets/%s_%s.txt'];
VOCopts.clsrespath=[VOCopts.resdir '%s_cls_' VOCopts.testset '_%s.txt'];
VOCopts.detrespath=[VOCopts.resdir 'det_%s_%s.txt'];


% initialize the VOC challenge options

% classes

VOCopts.classes={...
    'Pedestrian'
    };

VOCopts.nclasses=length(VOCopts.classes);	

% poses

VOCopts.poses={...
    'Unspecified'
    'Left'
    'Right'
    'Frontal'
    'Rear'};

VOCopts.nposes=length(VOCopts.poses);


% overlap threshold

VOCopts.minoverlap=0.5;

% annotation cache for evaluation

VOCopts.annocachepath=[VOCopts.localdir '%s_anno.mat'];

% options for example implementations

VOCopts.exfdpath=[VOCopts.localdir '%s_fd.mat'];

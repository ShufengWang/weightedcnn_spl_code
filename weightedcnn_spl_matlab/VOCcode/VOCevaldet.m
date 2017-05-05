function [rec,prec,ap] = VOCevaldet(VOCopts,id,cls,draw)

% load test set

cp=sprintf(VOCopts.annocachepath,VOCopts.testset);
if exist(cp,'file')
    fprintf('%s: pr: loading ground truth\n',cls);
    load(cp,'gtids','recs');
else
    [gtids,t]=textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s %d');
    for i=1:length(gtids)
        % display progress
        if toc>1
            fprintf('%s: pr: load: %d/%d\n',cls,i,length(gtids));
            drawnow;
            tic;
        end
        % read annotation
         recs{i}=PASreadrecord(sprintf(VOCopts.annopath,VOCopts.testset, gtids{i}));
    end
    save(cp,'gtids','recs');
end

fprintf('%s: pr: evaluating detections\n',cls);

% hash image ids
hash=VOChash_init(gtids);
        
% extract ground truth objects

npos=0;
gt(length(gtids))=struct('BB',[],'diff',[],'det',[]);
for i=1:length(gtids)
    % extract objects of class
    if isfield(recs{i}, 'objects')
        clsinds=strmatch(cls,{recs{i}.objects(:).class},'exact');
        gt(i).BB=cat(1,recs{i}.objects(clsinds).bbox)';
        gt(i).diff=[recs{i}.objects(clsinds).difficult];
        gt(i).det=false(length(clsinds),1);
        npos=npos+sum(~gt(i).diff);
    else
        gt(i).BB = [];
        gt(i).diff = [];
        gt(i).det=[];
    end
end

% load results
[ids,confidence,b1,b2,b3,b4]=textread(sprintf(VOCopts.detrespath,VOCopts.testset,cls),'%s %f %f %f %f %f');
BB=[b1 b2 b3 b4]';

% sort detections by decreasing confidence
[sc,si]=sort(-confidence);
ids=ids(si);
BB=BB(:,si);

% assign detections to ground truth objects
nd=length(confidence);
tp=zeros(nd,1);
fp=zeros(nd,1);
tic;
for d=1:nd
    % display progress
    if toc>1
        fprintf('%s: pr: compute: %d/%d\n',cls,d,nd);
        drawnow;
        tic;
    end
    
    % find ground truth image
    i=VOChash_lookup(hash,ids{d});
    if isempty(i)
        error('unrecognized image "%s"',ids{d});
    elseif length(i)>1
        error('multiple image "%s"',ids{d});
    end

    % assign detection to ground truth object if any
    bb=BB(:,d);
    ovmax=-inf;
    for j=1:size(gt(i).BB,2)
        bbgt=gt(i).BB(:,j);
        bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 & ih>0                
            % compute overlap as area of intersection / area of union
            ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
               (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
               iw*ih;
            ov=iw*ih/ua;
            if ov>ovmax
                ovmax=ov;
                jmax=j;
            end
        end
    end
    % assign detection as true positive/don't care/false positive
    if ovmax>=VOCopts.minoverlap
        if ~gt(i).diff(jmax)
            if ~gt(i).det(jmax)
                tp(d)=1;            % true positive
                gt(i).det(jmax)=true;
            else
                fp(d)=1;            % false positive (multiple detection)
            end
        end
    else
        fp(d)=1;                    % false positive
    end
end

% compute precision/recall
fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/npos;
prec=tp./(fp+tp);

fppi = fp/length(gtids);
misr = 1-rec;
det = [misr,fppi];
det = det(1:2:end,:);
fid = fopen([VOCopts.datadir 'results/det_' VOCopts.testset '.txt'],'w');
for i=1:size(det,1)
    fprintf(fid,'%.3f   %.3f\n',det(i,1),det(i,2));
    if det(i,2) < 1
        thres_fppi1 = -sc(2*i-1);
        save([VOCopts.datadir 'results/thres_fppi1.mat'], 'thres_fppi1');
    end
end
fclose(fid);

ap=VOCap(rec,prec);

if draw
    % plot precision/recall
    plot(rec,prec,'-');
    grid;
    xlabel 'recall'
    ylabel 'precision'
    title(sprintf('class: %s, subset: %s, AP = %.3f',cls,VOCopts.testset,ap));
end

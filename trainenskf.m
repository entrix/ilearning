function [out1,out2] = trainenskf(varargin)
%TRAINENSKF Ensemble Kalman filter implementation.
%
%  <a href="matlab:doc trainlm">trainlm</a> is a network training function that updates weight and
%  bias states according to Levenberg-Marquardt optimization.
%
%  <a href="matlab:doc trainlm">trainlm</a> is often the fastest backpropagation algorithm in the toolbox,
%  and is highly recommended as a first choice supervised algorithm,
%  although it does require more memory than other algorithms.
%
%  [NET,TR] = <a href="matlab:doc trainlm">trainlm</a>(NET,X,T) takes a network NET, input data X
%  and target data T and returns the network after training it, and a
%  a training record TR.
%  
%  [NET,TR] = <a href="matlab:doc trainlm">trainlm</a>(NET,X,T,Xi,Ai,EW) takes additional optional
%  arguments suitable for training dynamic networks and training with
%  error weights.  Xi and Ai are the initial input and layer delays states
%  respectively and EW defines error weights used to indicate
%  the relative importance of each target value.
%
%  Training occurs according to training parameters, with default values.
%  Any or all of these can be overridden with parameter name/value argument
%  pairs appended to the input argument list, or by appending a structure
%  argument with fields having one or more of these names.
%    show        25  Epochs between displays
%    showCommandLine 0 generate command line output
%    showWindow   1 show training GUI
%    epochs     100  Maximum number of epochs to train
%    goal         0  Performance goal
%    max_fail     5  Maximum validation failures
%    min_grad 1e-10  Minimum performance gradient
%    mu       0.001  Initial Mu
%    mu_dec     0.1  Mu decrease factor
%    mu_inc      10  Mu increase factor
%    mu_max    1e10  Maximum Mu
%    time       inf  Maximum time to train in seconds
%
%  To make this the default training function for a network, and view
%  and/or change parameter settings, use these two properties:
%
%    net.<a href="matlab:doc nnproperty.net_trainFcn">trainFcn</a> = '<a href="matlab:doc trainlm">trainlm</a>';
%    net.<a href="matlab:doc nnproperty.net_trainParam">trainParam</a>
%
%  See also trainscg, feedforwardnet, narxnet.

% Mark Beale, 11-31-97, ODJ 11/20/98
% Updated by Orlando De Jes�s, Martin Hagan, Dynamic Training 7-20-05
% Copyright 1992-2012 The MathWorks, Inc.
% $Revision: 1.1.6.18 $ $Date: 2012/08/21 01:05:22 $

%% =======================================================
%  BOILERPLATE_START
%  This code is the same for all Training Functions.

  persistent INFO;
  if isempty(INFO), INFO = get_info; end
  nnassert.minargs(nargin,1);
  in1 = varargin{1};
  if ischar(in1)
    switch (in1)
      case 'info'
        out1 = INFO;
      case 'apply'
        [out1,out2] = train_network(varargin{2:end});
      case 'formatNet'
        out1 = formatNet(varargin{2});
      case 'check_param'
        param = varargin{2};
        err = nntest.param(INFO.parameters,param);
        if isempty(err)
          err = check_param(param);
        end
        if nargout > 0
          out1 = err;
        elseif ~isempty(err)
          nnerr.throw('Type',err);
        end
      otherwise,
        try
          out1 = eval(['INFO.' in1]);
        catch me, nnerr.throw(['Unrecognized first argument: ''' in1 ''''])
        end
    end
  else
    net = varargin{1};
    oldTrainFcn = net.trainFcn;
    oldTrainParam = net.trainParam;
    if ~strcmp(net.trainFcn,mfilename)
      net.trainFcn = mfilename;
      net.trainParam = INFO.defaultParam;
    end
    [out1,out2] = train(net,varargin{2:end});
    net.trainFcn = oldTrainFcn;
    net.trainParam = oldTrainParam;
  end
end

%  BOILERPLATE_END
%% =======================================================

function info = get_info()
  isSupervised = true;
  usesGradient = false;
  usesJacobian = true;
  usesValidation = true;
  supportsCalcModes = true;
  info = nnfcnTraining(mfilename,'Levenberg-Marquardt',8.0,...
    isSupervised,usesGradient,usesJacobian,usesValidation,supportsCalcModes,...
    [ ...
    nnetParamInfo('showWindow','Show Training Window Feedback','nntype.bool_scalar',true,...
    'Display training window during training.'), ...
    nnetParamInfo('showCommandLine','Show Command Line Feedback','nntype.bool_scalar',false,...
    'Generate command line output during training.'), ...
    nnetParamInfo('show','Command Line Frequency','nntype.strict_pos_int_inf_scalar',25,...
    'Frequency to update command line.'), ...
    ...
    nnetParamInfo('epochs','Maximum Epochs','nntype.pos_int_scalar',1000,...
    'Maximum number of training iterations before training is stopped.'), ...
    nnetParamInfo('time','Maximum Training Time','nntype.pos_inf_scalar',inf,...
    'Maximum time in seconds before training is stopped.'), ...
    ...
    nnetParamInfo('goal','Performance Goal','nntype.pos_scalar',0,...
    'Performance goal.'), ...
    nnetParamInfo('min_grad','Minimum Gradient','nntype.pos_scalar',1e-7,...
    'Minimum performance gradient before training is stopped.'), ...
    nnetParamInfo('max_fail','Maximum Validation Checks','nntype.strict_pos_int_scalar',6,...
    'Maximum number of validation checks before training is stopped.'), ...
    ...
    nnetParamInfo('mu','Mu','nntype.pos_scalar',0.001,...
    'Mu.'), ...
    nnetParamInfo('mu_dec','Mu Decrease Ratio','nntype.real_0_to_1',0.1,...
    'Ratio to decrease mu.'), ...
    nnetParamInfo('mu_inc','Mu Increase Ratio','nntype.over1',10,...
    'Ratio to increase mu.'), ...
    nnetParamInfo('mu_max','Maximum mu','nntype.strict_pos_scalar',1e10,...
    'Maximum mu before training is stopped.'), ...
    ], ...
    [ ...
    nntraining.state_info('gradient','Gradient','continuous','log') ...
    nntraining.state_info('mu','Mu','continuous','log') ...
    nntraining.state_info('val_fail','Validation Checks','discrete','linear') ...
    ]);
end

function err = check_param(param)
  err = '';
end

function net = formatNet(net)
  if isempty(net.performFcn)
    warning('nnet:trainlm:Performance',nnwarning.empty_performfcn_corrected);
    net.performFcn = 'mse';
    net.performParam = mse('defaultParam');
  end
  if isempty(nnstring.first_match(net.performFcn,{'sse','mse'}))
    warning('nnet:trainlm:Performance',nnwarning.nonjacobian_performfcn_replaced);
    net.performFcn = 'mse';
    net.performParam = mse('defaultParam');
  end
end

function [calcNet,tr] = train_network(archNet,rawData,calcLib,calcNet,tr)
  
  % Parallel Workers
  isParallel = calcLib.isParallel;
  isMainWorker = calcLib.isMainWorker;
  mainWorkerInd = calcLib.mainWorkerInd;

  % Create broadcast variables
  stop = [];
  
  % Initialize
%   archNet.trainParam.showWindow = false;
  param = archNet.trainParam;
  if isMainWorker
    startTime = clock;
    original_net = calcNet;
  end
  
  [perf,vperf,tperf,~,jj,gradient] = calcLib.perfsJEJJ(calcNet);
  
  if isMainWorker
    [best,val_fail] = nntraining.validation_start(calcNet,perf,vperf);
    WB = calcLib.getwb(calcNet);
    lengthWB = length(WB);
%     ii = sparse(1:lengthWB,1:lengthWB,ones(1,lengthWB));
    mu = param.mu;
    % set as general net
    net = archNet;

    % Training Record
    tr.best_epoch = 0;
    tr.goal = param.goal;
    tr.states = {'epoch','time','perf','vperf','tperf','gradient','val_fail'};

    % Status
    status = ...
      [ ...
      nntraining.status('Epoch','iterations','linear','discrete',0,param.epochs,0), ...
      nntraining.status('Time','seconds','linear','discrete',0,param.time,0), ...
      nntraining.status('Performance','','log','continuous',perf,param.goal,perf) ...
      nntraining.status('Gradient','','log','continuous',gradient,param.min_grad,gradient) ...
      nntraining.status('Validation Checks','','linear','discrete',0,param.max_fail,0) ...
      ];
    nn_train_feedback('start',archNet,status);
  end

  % get external parameters
  WB         = getwb(net);
  lengthWB   = length(WB);
  
  e             = get_def_field(net, 'e', 1e-2);
  nu            = get_def_field(net, 'nu', 5e-1);
  q             = get_def_field(net, 'q', 2);
  norm_max      = get_def_field(net, 'norm_max', 1e3);
  isIncremental = get_def_field(net, 'isIncremental', false);
  toDiagonal    = get_def_field(net, 'toDiagonal', false);
  normalize     = get_def_field(net, 'normalize', false);
  
  
  X   = rawData.X;
  PDc = rawData.Xi;
  Ai  = rawData.Ai;
  T   = rawData.T;
  Tmat = cell2mat(T);
  P = [];
  
  R = (1 / nu) * eye( numel(T) );
  Q = q        * eye( lengthWB );
  %   q  = net.userdata.q;
  
  
  % ensemble parameter
  n = get_def_field(net, 'n', 10);
  
  % net output parameters
  v_out_len   = numel(T);
  v_train_len = n;
  
  
  WB_list = zeros(lengthWB, v_train_len);
  Y_list  = zeros(v_out_len, v_train_len);
  
  min_WB = min(min(WB));
  max_WB = max(max(WB));
  
  % initialize ensemble weights
  for i = 1:10
    WB_list(:, i) = min_WB + (max_WB - min_WB).*rand(lengthWB,1);
  end
  
  % initialize ensemble outs
  for i = 2:n
    net = setwb(net, WB_list(:, i));
    Y_list(:, i) = cell2mat(sim(network(net), X, PDc, Ai, T));
  end
  y_mean = (1 / n) * sum(Y_list(:,:)')';
  Ey = [ Y_list - y_mean(:, ones(1,v_train_len)) ];
  
  % stored parameters
  if get_def_field(net, 'printMatrix', false)
    loop_n = net.userdata.loop_n;
  end
  if get_def_field(net, 'P', [])
      P = net.userdata.P;
  end
  if (isempty(P))
    P = (1 / e)  * eye( v_out_len );
  end
  
  % Train
  for epoch = 0:param.epochs
    
    % Stopping Criteria
    if isMainWorker
      current_time = etime(clock,startTime);
      [userStop,userCancel] = nntraintool('check');
      if userStop, tr.stop = 'User stop.'; calcNet = best.net;
      elseif userCancel, tr.stop = 'User cancel.'; calcNet = original_net;
      elseif (perf <= param.goal), tr.stop = 'Performance goal met.'; calcNet = best.net;
      elseif (epoch == param.epochs), tr.stop = 'Maximum epoch reached.'; calcNet = best.net;
      elseif (current_time >= param.time), tr.stop = 'Maximum time elapsed.'; calcNet = best.net;
      elseif (gradient <= param.min_grad), tr.stop = 'Minimum gradient reached.'; calcNet = best.net;
        %       elseif (mu >= param.mu_max), tr.stop = 'Maximum MU reached.'; calcNet = best.net;
      elseif (val_fail >= param.max_fail), tr.stop = 'Validation stop.'; calcNet = best.net;
      end
      
      % Feedback
      tr = nntraining.tr_update(tr,[epoch current_time perf vperf tperf mu gradient val_fail]);
      statusValues = [epoch,current_time,best.perf,gradient,mu,val_fail];
      nn_train_feedback('update',archNet,rawData,calcLib,calcNet,tr,status,statusValues);
      stop = ~isempty(tr.stop);
    end
    
    % Stop
    if isParallel, stop = labBroadcast(mainWorkerInd,stop); end
    if stop, break, end

    % Ensemble Kalman Filter
    % **********************************************

    % recalculate inner loop variables
    WB_mean = (1 / n) * sum(WB_list(:,:)')';
    E = [ WB_list - WB_mean(:, ones(1,v_train_len)) ];
    
    Pxy = (1 / (n - 1)) * E  * Ey';
    Pyy = P * (1 / (n - 1)) * Ey * Ey' * P + R;
    
%     Pxy = toDiag(Pxy);
%     Pyy = toDiag(Pyy);
    
%     if normalize
%       normP = max(max(Pxy));
%       if(normP > 5*norm_max),
%         Pxy = Pxy.*norm_max/normP;
%       end
%       normP = max(max(Pyy));
%       if(normP > 5*norm_max),
%         Pyy = Pyy.*norm_max/normP;
%       end
%     end

    if (toDiagonal)
      Pxy = toDiag(Pxy);
      Pyy = toDiag(Pyy);
    end
    
    K   = Pxy/Pyy;

    y_mean = (1 / n) * sum(Y_list(:,:)')';
    Ey = [ Y_list - y_mean(:, ones(1,v_train_len))  ];

    % calculate ensemble outs
    for i = 1:n
      net = setwb(net, WB_list(:, i));
      Y_list(:, i) = cell2mat(sim(network(net), X, PDc, Ai, T));
    end
    
    % calculate ensemble eeights
    for i = 1:n
      net = setwb(net, WB_list(:, i));
      WB_list(:, i) = WB_list(:, i) +...
        K * (Tmat(:) + R(:, i) - Y_list(:, i)) + Q(:, i);
    end
    % **********************************************
    
    % check results
    if (get_def_field(net, 'checkIter', false))
      check_iter();
    end
    
    calcNet = calcLib.setwb(calcNet,WB_mean);
    
    % Validation
    [perf,vperf,tperf,~,~,gradient] = calcLib.perfsJEJJ(calcNet);
    if isMainWorker
      [best,tr,val_fail] = nntraining.validation(best,tr,val_fail,calcNet,perf,vperf,epoch);
    end
  end
  
  % save parameters for follow invocations
  calcNet.userdata.P = P;
  
  function check_iter
    % check performance
    net = setwb(net,WB_mean);
    Yts = sim(network(net), X, PDc, Ai, T);
    perf2 = feval(net.performFcn,net,Yts,T);
    
    fprintf('\titer = %d\t%s before = %f\t%s after = %f\tP_max = %f\n',...
      epoch, net.performFcn, perf, net.performFcn, perf2, max(max(abs(P))));
    
    % print matrix visualization to file
    if (get_def_field(net, 'printMatrix', false))
      if (exist('trainenskf','dir') ~= 7)
        mkdir 'trainenskf';
      end
      figure1 = figure;
      axes1 = axes('Parent',figure1);
      hold(axes1,'all');
      set(figure1,'Visible', 'off');
      imagesc(P);            %# Create a colored plot of the matrix values
      colormap(bone);
      saveas(figure1,sprintf('trainenskf\\cov_matrix-%d-%d.png', loop_n, epoch))  % here you save the figure
      close(figure1);
      % view 3D visualization of the matrix values
      %       [x, y] = stemMat(P);
      %       stem3(x,y,P,'MarkerFaceColor','g');
    end
  end
end


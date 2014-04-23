function [out1,out2,out3] = adaptekf(in1,in2,in3,in4,in5,in6,in7)
%ADAPTEKF Sequential order incremental adaption w/learning functions
%  through extended Kalman filter implementation.

%  [NET,AR,AC] = <a href="matlab:doc adaptwb">adaptwb</a>(NET,X,PD,T,AI) takes a network, delayed inputs,
%  targets, and initial layer states, and returns the updated network,
%  adaption record, and layer outputs after applying the network's
%  weight and bias learning rules for each timestep in T.
%
%  <a href="matlab:doc adaptwb">adaptwb</a> is not commonly called directly, it is called by ADAPT when
%  the network's adaption function net.<a href="matlab:doc nnproperty.net_adaptFcn">adaptFcn</a> is set to <a href="matlab:doc adaptwb">adaptwb</a>.

% Mark Beale, 11-31-97
% Copyright 1992-2012 The MathWorks, Inc.
% $Revision: 1.1.10.8 $  $Date: 2012/08/21 01:03:31 $

% TODO - Replace PD with Xc, TAPDELAY, return Yc?

%% =======================================================
%  BOILERPLATE_START
%  This code is the same for all Adapt Functions.

  persistent INFO;
  if isempty(INFO), INFO = get_info; end
  if (nargin < 1), error(message('nnet:Args:NotEnough')); end
  if ischar(in1)
    switch (in1)
      case 'info'
        out1 = INFO;
      case 'check_param'
        out1 = check_param(in2);
      otherwise,
        try
          out1 = eval(['INFO.' in1]);
        catch me, nnerr.throw(['Unrecognized first argument: ''' in1 ''''])
        end
    end
  else
    [out1,out2,out3] = adapt_network(in1,in2,in3,in4,in5);
  end
end

%  BOILERPLATE_END
%% =======================================================

function [net,Ac,tr] = adapt_network(net,X,PD,T,Ai)
  TS = numtimesteps(T);

  %Signals
  Ac = [Ai cell(net.numLayers,TS)];
    
  % initialize
  tr.timesteps = 1:TS;
  tr.perf = zeros(1,TS);
  
  % get external parameters
  WB = getwb(net);
  lengthWB = length(WB);
  
  e             = get_def_field(net, 'e', 1e-2);
  nu            = get_def_field(net, 'nu', 5e-1);
  q             = get_def_field(net, 'q', 2);
  norm_max      = get_def_field(net, 'norm_max', 1e3);
  isIncremental = get_def_field(net, 'isIncremental', false);
  toDiagonal    = get_def_field(net, 'toDiagonal', false);
  normalize     = get_def_field(net, 'normalize', false);
  
  P = [];
  R = (1 / nu) * eye( lengthWB );
  Q = q        * eye( lengthWB );
  
  % stored parameters
  if get_def_field(net, 'printMatrix', false)
    loop_n = net.userdata.loop_n;
  end
  if get_def_field(net, 'P', [])
      P = net.userdata.P;
  end
  if (isempty(P))
    P = (1 / e)  * eye( lengthWB );
  end
  
  % Adapt
  for ts=1:TS
    % Extended Kalman Filter
    % **********************************************   

    %     while true
    ind = mod(int8(rand*100),99);
    while ind == 0
      ind = mod(int8(rand*100),99);
    end
    [tr.perf(ts),Ewb,Jx,~] = calcJeJx(net, X(ts), PD(1,:,ts), Ai, T(ts), {1});
    
    if ~isIncremental
        P = (1 / e)  * eye( lengthWB );
    elseif toDiagonal
        P = toDiag(P);
    end
    
    %             Q = eye( calcHints )*1e-6;
    H = -Jx';
    
    if normalize
      normP = max(max(P));
      if(normP > 5*norm_max),
        P = P.*norm_max/normP;
      end
    end
    
    % dbstop if warning
    I = eye( lengthWB );
    
    P = P + Q;
    S = H*P*H' + R;
    K = P*H'/S;
    P2 = (I-K*H)*P*(I-K*H)'+K*R*K';
    WB2 = WB + K*Ewb;
    
    PDc = cellMat(PD(1,:,ts), net.numInputDelays);
    if (get_def_field(net, 'checkIter', false))
      check_iter();
    end
    
    P = P2;
    WB = WB2;
    
%     fprintf('\titer = %d\tmse  = %f\n', ts, mse(net, cell2mat(T(ts)), Y));
    net = setwb(net, WB);
    % **********************************************
  end
  
  % save parameters for follow invocations
  net.userdata.P = P;
%   net = setwb(net, WB);
  
  function check_iter
    % check performance
    net = setwb(net,WB);
    Yts = sim(network(net), X(ts), PDc, Ai, T(ts));
    perf = feval(net.performFcn,net,Yts,cell2mat(T(ts)));
    net = setwb(net,WB2);
    Yts = sim(network(net), X(ts), PDc, Ai, T(ts));
    perf2 = feval(net.performFcn,net,Yts,cell2mat(T(ts)));
    Y = calcY(net, X, PD(1,:,:), Ai, T, {1});
    Y = Y{1};
    perf3 = feval(net.performFcn,net,Y,cell2mat(T));
    
    fprintf('\titer = %d\t%s before = %f\t%s after = %f\t%s full = %f\tP_max = %f\n',...
      ts, net.performFcn, perf, net.performFcn, perf2, net.performFcn, perf3, max(max(abs(P))));
    
    % print matrix visualization to file
    if (get_def_field(net, 'printMatrix', false))
      if (exist('adaptekf','dir') ~= 7)
        mkdir 'adaptekf';
      end
      figure1 = figure;
      axes1 = axes('Parent',figure1);
      hold(axes1,'all');
      set(figure1,'Visible', 'off');
      imagesc(P);            %# Create a colored plot of the matrix values
      colormap(bone);
      saveas(figure1,sprintf('adaptekf\\cov_matrix-%d-%d.png', loop_n, ts))  % here you save the figure
      close(figure1);
      % view 3D visualization of the matrix values
      %       [x, y] = stemMat(P);
      %       stem3(x,y,P,'MarkerFaceColor','g');
    end
  end
end

function v = fcnversion
  v = 7;
end

%===============================================================

function info = get_info
  info = nnfcnAdaptive(mfilename,'Weight/Bias Rule Adaption',fcnversion,...
    true,true,[]);
end

function err = check_param(param)
  err = '';
end

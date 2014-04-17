function [out1,out2,out3,out4] = adaptekf_ms(in1,in2,in3,in4,in5,in6,in7)
  %ADAPTWB Sequential order incremental adaption w/learning functions.
  %
  %  [NET,AR,AC] = <a href="matlab:doc adaptwb">adaptwb</a>(NET,PD,T,AI) takes a network, delayed inputs,
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
    [out1,out2,out3,out4] = adapt_network(in1,in2,in3,in4,in5);
  end
end

%  BOILERPLATE_END
%% =======================================================

function [net,Ac,tr,WB_list] = adapt_network(net,X,PD,T,Ai)
  Q = numsamples(T);
  TS = numtimesteps(T);

  % Calculation Mode
  calcMode = nncalc.defaultMode(net,Ai);
  %   calcMode.options = nnet.options.calc.defaults;
  calcMode.options.showResources = 'yes';

  % Constants
  %   numLayers = net.numLayers;
  %   numInputs = net.numInputs;
  %   performFcn = net.performFcn;
  %   performParam = net.performParam;
  %   needGradient = nn.needsGradient(net);
  %   numLayerDelays = net.numLayerDelays;

  %Signals

  Ac = [Ai cell(net.numLayers,TS)];

  % Initialize
  tr.timesteps = 1:TS;
  tr.perf = zeros(1,TS);

  WB = getwb(net);
  lengthWB = length(WB);

  e = net.userdata.e;
  nu = net.userdata.nu;
  q = net.userdata.q;
  norm_max = net.userdata.norm_max;
  isIncremental = net.userdata.isIncremental;
  toDiagonal = net.userdata.toDiagonal;
  % multistream parameters
  l = net.userdata.l;
  g = net.userdata.g;

%   P = (1 / e)  * eye( lengthWB );
  R = (1 / nu) * eye( lengthWB * g );
  Q = q        * eye( lengthWB );
  WB_list      = cell(1,g);
  H_con_list   = cell(1,g);
  Ewb_con_list = cell(1,g);
  K_list       = cell(1,g);
  P_list       = cell(1,g);

  skipUpdate = true;

  % initialize weights for each stream
  for i = 1:g
    WB_list{i} = WB;
    P_list{i}  = (1 / e)  * eye( lengthWB );
  end
  % Adapt
  for ts = 1:TS
    if mod(ts,l) == 0
      skipUpdate = false;
      % prepare global matrix A
      S = R;
      for i = 1:g
        S = S + H_con_list{i}'*P_list{1}*H_con_list{i};
      end
    end
    for i = 1:g
      if skipUpdate
        % ret random instance
        ind = mod(int8(rand*100),99);
        while ind == 0
          ind = mod(int8(rand*100),99);  
        end
        % set weight for i-th group
        setwb(net, WB_list{i});
        % Kalman Filter
        % **********************************************
        [tr.perf(ts),Ewb,Jx,Y] = calcJeJx(net, X(ind), PD(1,:,ind), Ai, T(ind), {1});

        if ~isIncremental
          P_list{i} = (1 / e)  * eye( lengthWB );
        elseif toDiagonal
          P_list{i} = toDiag(P_list{i});
        end

        % concatenate the derivative matrices for i-th group
        H_con_list{i} = cat(2, H_con_list{i}, -Jx');
        % concatenate error vector for i-th group
        Ewb_con_list{i} = cat(2, Ewb_con_list{i}, Ewb');
      else
        normP = max(max(P_list{i}));
        if(normP > 5*norm_max),
          P_list{i} = P_list{i}.*norm_max/normP;
        end
        % dbstop if warning
        I = eye( lengthWB );
        P_list{i} = P_list{i} + Q;
        K_list{i} = P_list{i}*H_con_list{i}/S;
        P_list{i} = (I-K_list{i}*H_con_list{i}')*P_list{i}*(I-K_list{i}*H_con_list{i}')'+...
          K_list{i}*R*K_list{i}';
        WB_list{i} = WB_list{i} + K_list{i}*Ewb_con_list{i}';
        fprintf('\titer = %d\tgroup = %d\tmse  = %f\n', ts, i, mse(net, cell2mat(T(ts)), Y));
      end
      % **********************************************
    end
    if ~skipUpdate
      % clear matrices
        H_con_list   = cell(1,g);
        Ewb_con_list = cell(1,g);
    end
    skipUpdate = true;
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

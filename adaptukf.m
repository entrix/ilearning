function [out1,out2,out3] = adaptukf(in1,in2,in3,in4,in5,in6,in7)
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
    [out1,out2,out3] = adapt_network(in1,in2,in3,in4,in5);
  end
end

%  BOILERPLATE_END
%% =======================================================

function [net,Ac,tr] = adapt_network(net,X,PD,T,Ai)
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
  
  e  = net.userdata.e;
  nu = net.userdata.nu;
  q  = net.userdata.q;
  L  = numel(WB);
  alpha  = net.userdata.alpha;
  beta   = net.userdata.beta;
  k  = 3 - L;
  lambda = alpha^2 * (L + k) - L;
  W_m_0  = lambda / (L + lambda);
  W_c_0  = lambda / (L + lambda) + 1 - alpha^2 + beta;
  W_mc_i = 1 /  (2 * (L + lambda));
  gamma  = sqrt(L + lambda);
  lambda_RLS = net.userdata.lambda_RLS;
  
  P =   (1 / e)  * eye( lengthWB );
  R_e = 0;%(1 / nu) * eye( length(T(1)) );
  R_r = q        * eye( lengthWB );
%   WB_sigma = cell(1,2 * L + 1);
    D_sigma = zeros(length(T(1)), numel(T));
  % Adapt
  for ts=1:TS
    d = zeros(length(T(1)),1);
    P_d_d = zeros( length(T(1)), length(T(1)) );
    P_w_d = zeros( lengthWB,     length(T(1)) );

    % Unscented Kalman Filter
    % **********************************************   
%     R_r = (1 / lambda_RLS - 1) * P;
    P_w_min = P + R_r;
    
%     sq = sqrtm(P_w_min);
    WB_sigma = sigmas(WB,P_w_min,gamma);
    for i = 1:(2 * L + 1)
        setwb(net, WB_sigma(:, i));
        D_sigma(:, i) = cell2mat(calcY(net, X(ts), PD(1,:,ts), Ai, T(ts), {1}));
    end
    
    for i = 1:(2 * L + 1)
      if i == 1
        W = W_m_0;
      elseif i <= L + 1
        W = W_mc_i;
      else
        W = W_mc_i;
      end
        d = d + W * D_sigma(:, i);
    end
    
    for i = 1:(2 * L + 1)
      if i == 1
        W = W_c_0;
      elseif i <= L + 1
        W = W_mc_i;
      else
        W = W_mc_i;
      end
      P_d_d = P_d_d + W * (D_sigma (:, i) - d)  * (D_sigma(:, i) - d)' + R_e;
      P_w_d = P_w_d + W * (WB_sigma(:, i) - WB) * (D_sigma(:, i) - d)';
    end
    
    K = P_w_d*(1/P_d_d);
    WB = WB + K * (cell2mat(T(ts)) - d);
    P  = P_w_min - K * P_d_d * K';
    
    avg = zeros(length(T(1)), 1);
    for i = 1:(2 * L + 1)
      avg = avg + D_sigma(:, i);
    end
    avg = avg / (2 * L + 1);
    fprintf('\titer = %d\tmse  = %f\n', ts, mse(net, cell2mat(T(ts)), avg));
    % **********************************************
  end
  net = setwb(net, WB);
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

function WB_sigma = sigmas(WB,P,gamma)
%Sigma points around reference point
%Inputs:
%       x: reference point
%       P: covariance
%       c: coefficient
%Output:
%       X: Sigma points

A = gamma*chol(P)';
Y = WB(:,ones(1,numel(WB)));
WB_sigma = [WB Y+A Y-A];
end

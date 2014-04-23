function [out1,out2,out3] = adaptukf(in1,in2,in3,in4,in5,in6,in7)
%ADAPTUKF Sequential order incremental adaption w/learning functions
%  through unscented Kalman filter implementation.
%
%  [NET,AR,AC] = <a href="matlab:doc adaptwb">adaptukf</a>(NET,X,PD,T,AI) takes a network, delayed inputs,
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
  % signals
  TS = numtimesteps(T);
  Ac = [Ai cell(net.numLayers,TS)];
    
  % initialize
  tr.timesteps = 1:TS;
  tr.perf = zeros(1,TS);

  % get external parameters
  WB         = getwb(net);
  lengthWB   = length(WB);
  
  e          = get_def_field(net, 'e', 1e-2);
  nu         = get_def_field(net, 'nu', 5e-1);
  alpha      = get_def_field(net, 'alpha', 1e-3);
  beta       = get_def_field(net, 'beta', 2);
  q          = get_def_field(net, 'q', 2);
  lambda_RLS = get_def_field(net, 'lambda_RLS', 1);
  alpha_RM   = get_def_field(net, 'alpha_RM', 5e-1);
  norm_max      = get_def_field(net, 'norm_max', 1e3);
  isIncremental = get_def_field(net, 'isIncremental', false);
  toDiagonal    = get_def_field(net, 'toDiagonal', false);
  normalize     = get_def_field(net, 'normalize', false);
  %   q  = net.userdata.q;

  % net output parameters
  v_out_len   = length(cell2mat(T(1)));
  v_train_len = 2 * numel(WB) + 1;
  
  % get inferential paramreters
  L  = numel(WB);
  k  = 3 - L;
  lambda = alpha^2 * (L + k) - L;
  W_m_0  = lambda / (L + lambda);
  W_c_0  = lambda / (L + lambda) + 1 - alpha^2 + beta;
  W_mc_i = 1 /  (2 * (L + lambda));
  gamma  = sqrt(L + lambda);
  R_r = q * eye( lengthWB );
  R_e = (1 / nu) * eye( v_out_len );
  P = [];
  
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
    % init inner loop variables
%     d = zeros(v_out_len,1);
    P_d_d = zeros( v_out_len, v_out_len );
    P_w_d = zeros( lengthWB,  v_out_len );
    
    % Unscented Kalman Filter
    % **********************************************
    %     R_r = (1 / lambda_RLS - 1) * P;
    
    if ~isIncremental
      P = (1 / e)  * eye( lengthWB );
    end
    
    if normalize
      normP = max(max(P));
      if(normP > 5*norm_max),
        P = P.*norm_max/normP;
      end
    end
      
    P_w_min = P + R_r;
    
    if (toDiagonal)
      P_w_min = toDiag(P_w_min);
    end
    
    % calculate sigma outs
    WB_sigma = sigmas(WB,P_w_min,gamma);
    
    % calculate sigma outs
    D_sigma = zeros(v_out_len, v_train_len);
    PDc = cellMat(PD(1,:,ts), net.numInputDelays);
    for i = 1:(2 * L + 1)
      net = setwb(net, WB_sigma(:, i));
      D_sigma(:, i) = cell2mat(sim(network(net), X(ts), PDc, Ai, T(ts)));
      %       D_sigma(:, i) = cell2mat(calcY(net, X(ts), PD(1,:,ts), Ai, T(ts), {1}));
    end
    
    % calculate net out
    %     for i = 1:(2 * L + 1)
    %       if i == 1
    %         W = W_m_0;
    %       else
    %         W = W_mc_i;
    %       end
    %         d = d + W * D_sigma(:, i);
    %     end
    %     d = d / (2*L + 1);
    % set parameter as well as in EKF
    d = D_sigma(:, 1);
    
    % calculate Kalman gain
    for i = 1:(2 * L + 1)
      if i == 1
        W = W_c_0;
      else
        W = W_mc_i;
      end
      P_d_d = P_d_d + W * (D_sigma (:, i) - d)  * (D_sigma(:, i) - d)' + R_e;
      P_w_d = P_w_d + W * (WB_sigma(:, i) - WB) * (D_sigma(:, i) - d)';
    end
    K = P_w_d/P_d_d;
    
    % update weights
    WB2 = WB + K * (cell2mat(T(ts))' - d);
    
    % update covariance matrix
    P2  = P_w_min - K * P_d_d * K';
    
    % Robbin-Monro approximation
    R_r = (1 - alpha_RM) * R_r + alpha_RM * K * (cell2mat(T(ts))' - d) *...
      (cell2mat(T(ts))' - d)' * K';
    %     R_e = (1 - alpha_RM) * R_e + alpha_RM * (cell2mat(T(ts)) - d) *...
    %       (cell2mat(T(ts)) - d)';
    
    if (toDiagonal)
      % force matrices to diagonal form
      R_r = toDiag(R_r);
      %       R_e = toDiag(R_e);
    end
    % **********************************************
    
    % check results
    if (get_def_field(net, 'checkIter', false))
      check_iter();
    end
    
    % update values
    WB = WB2;
    P  = P2;
  end
  
  % save parameters for follow invocations
  net.userdata.P = P;
  net = setwb(net, WB);
  
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
      if (exist('adaptukf','dir') ~= 7)
        mkdir 'adaptukf';
      end
      figure1 = figure;
      axes1 = axes('Parent',figure1);
      hold(axes1,'all');
      set(figure1,'Visible', 'off');
      imagesc(P);            %# Create a colored plot of the matrix values
      colormap(bone);
      saveas(figure1,sprintf('adaptukf\\cov_matrix-%d-%d.png', loop_n, ts))  % here you save the figure
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

function [x,y] = stemMat(z)
%Representaion x and y matrices for stem3 functions
%Inputs:
%       z: matrix that will be visualized
%Output:
%       x: matrix for X axis
%       y: matrix for Y axis

  l = length(z);
  x = linspace(1,l,l)';
  x = x(:, ones(1, numel(x)));
  y = x';
end

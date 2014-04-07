function ekf_train_test
    net = timedelaynet(1:2, 10);
    % init input data
    [X,T] = simpleseries_dataset;
    % prepare input data
    [Xs,Xi,Ai,Ts] = preparets(net,X,T);
    % disable window
    net.trainParam.showWindow = false;
    % set our train function for EKF
    net.trainFcn = 'trainkf';
    % lets target data being accessible during training
    net.userdata.Ts = Ts;
    % init epochs
    net.userdata.epochs = 0;
    % start train
    [net,tr] = train(net,Xs,Ts,Xi,Ai);
    fprintf('                                                         \n');
    fprintf('\tnumber of epochs  = %d\n', length(tr.epoch));
    [Y,~,~] = sim(net,Xs,Xi,Ai,Ts);
    fprintf(strcat('\tmean square error = %f\n'), mse(net,Ts,Y));
    fprintf('                                                         \n');
    plot(cell2mat(Xs),cell2mat(Ts),'o',cell2mat(Xs),cell2mat(Y),'x')
end

classdef ESN < handle 
    properties
        Nr 
        alpha
        rho
        inputScaling
        biasScaling
        lambda
        connectivity
        readout_training
        Win
        Wb
        Wr
        Wout
        internalState
    end
    methods
        function esn = ESN(Nr, varargin)
            esn.Nr = Nr;
            esn.alpha = 1;
            esn.rho = 0.9;
            esn.inputScaling = 1;
            esn.biasScaling = 1;
            esn.lambda = 1;
            esn.connectivity = 1;
            esn.readout_training = 'ridgeregression';
            
            numvarargs = length(varargin);
            for i = 1:2:numvarargs
                switch varargin{i}
                    case 'leakRate', esn.alpha = varargin{i+1};
                    case 'spectralRadius', esn.rho = varargin{i+1};
                    case 'inputScaling', esn.inputScaling = varargin{i+1};
                    case 'biasScaling', esn.biasScaling = varargin{i+1};
                    case 'regularization', esn.lambda = varargin{i+1};
                    case 'connectivity', esn.connectivity = varargin{i+1};
                    case 'readoutTraining', esn.readout_training = varargin{i+1};
                    
                    otherwise, error('the option does not exist');
                end
            end
        end
        function train(esn, trX, trY, washout)    
            seqDim = size(trX{1},2);
            N = length(trX);
            trainLen = size(trY,2);
            
            esn.Win = esn.inputScaling * (rand(esn.Nr, size(trX{1},2)) * 2 - 1);
            esn.Wb = esn.biasScaling * (rand(esn.Nr, 1) * 2 - 1);
            esn.Wr = full(sprand(esn.Nr,esn.Nr, esn.connectivity));
            esn.Wr(esn.Wr ~= 0) = esn.Wr(esn.Wr ~= 0) * 2 - 1;
            esn.Wr = esn.Wr * (esn.rho / max(abs(eig(esn.Wr))));
            
            X = zeros(1+seqDim+esn.Nr, trainLen);
            idx = 1;
            for s = 1:N
                U = trX{s}';
                x = zeros(esn.Nr,1);
                for i = 1:size(U,2)
                    u = U(:,i);
                    x_ = tanh(esn.Win*u + esn.Wr*x + esn.Wb); 
                    x = (1-esn.alpha)*x + esn.alpha*x_;
                    if i > washout
                        X(:,idx) = [1;u;x];
                        idx = idx+1;
                    end
                end
            end
            esn.internalState = X(1+seqDim+1:end,:);
            esn.Wout = feval(esn.readout_training, X, trY, esn);
        end
        function y = predict(esn, data, washout)
            
            seqDim = size(data{1},2);
            N = length(data);
            trainLen = 0;
            for s = 1:N
                trainLen = trainLen + size(data{s},1) - washout;
            end
            
            X = zeros(1+seqDim+esn.Nr, trainLen);
            idx = 1;
            for s = 1:N
                U = data{s}';
                x = zeros(esn.Nr,1);
                
                for i = 1:size(U,2)
                    u = U(:,i);
                    x_ = tanh(esn.Win*u + esn.Wr*x + esn.Wb); 
                    x = (1-esn.alpha)*x + esn.alpha*x_;
                    if i > washout
                        X(:,idx) = [1;u;x];
                        idx = idx+1;
                    end
                end
            end
            
            esn.internalState = X(1+seqDim+1:end,:);
            y = esn.Wout*X;
            y = y';
        end
    end
end

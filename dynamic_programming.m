%% Preliminar objects
% There are
% - 7 states (S={0,1,...,6})
% - 2 possible actions (A={1,2} where 1: standard safety protocol and 2:
%   emergency safety protocol)
% Therefore, there are 2^7=128 possible policies

clear; clc; close all;

N = 6;
p_i = [1/4, 1/20];
p_r = 1/5;
lambda = 0.995;

rng(1234); % fixed random seed, for replicability of the code

% Compute the transition matrix associated to the standard safety protocol
P1 = zeros(7,7);
for i=0:6
    for j=0:6
        P1(i+1,j+1) = binopdf([0:6], i, p_i(1)) * binopdf([0:6]+j-i, N-i, p_r)';
    end
end

% Compute the transition matrix associated to the emergency safety protocol
P2 = zeros(7,7);
for i=0:6
    for j=0:6
        P2(i+1,j+1) = binopdf([0:6], i, p_i(2)) * binopdf([0:6]+j-i, N-i, p_r)';
    end
end

% Compute the reward matrix associated to the standard safety protocol
R1 = zeros(7,7);
for i=0:6
    for j=0:6
        R1(i+1,j+1) = 50*(exp(i/7)-1);
    end
end

% Compute the reward matrix associated to the emergency safety protocol
R2 = zeros(7,7);
for i=0:6
    for j=0:6
        R2(i+1,j+1) = (i~=0)*35*(exp((i-1)/7)-1);
    end
end

% Concatenate the two transition matrices
P = cat(3,P1,P2);
% Concatenate the two reward matrices
R = cat(3,R1,R2);

% NB: I computed P1, P2, R1, R2 explicitly (because these matrices are just
% 7x7, hence they are not so massive in terms of memory occupation);
% nonetheless, one could compute p(i,a,j) and r(i,a,j) on-the-fly with the
% following functions
%
p = @(i,a,j) binopdf([0:6], i-1, p_i(a)) * binopdf([0:6]+j-i, N-(i-1), pr)';
r_standard = @(i) 50*(exp(i/7)-1);
r_emergency = @(i) (i~=0)*35*(exp((i-1)/7)-1);
r = @(i,a,j) (a==1)*r_standard(i-1) + (a==2)*r_emergency(i-1);
%
% NB: in this last formula j (the new state) is not used: all columns of
% R1 and R2 respectively are equal


%-----------------------------------------------------------------------%
%% Find mu* with dynamic programming (Q-factor value iteration algorithm)

% Step 1
k = 1;
Q_old = zeros(7,2); % |S|*|A|
epsilon = 0.1;
J_infty_norm_list = []; % this list will contain the quantities max(J_new-J_old)

while true

    % Step 2 - update Q
    for i=1:7
        for a=1:2
            Q_new(i,a) = sum( P(i,:,a) .* ( R(i,:,a) + lambda.*max(Q_old,[],2)' ) );
            % NB: the command max(Q_old,[],2) means "find the maximum
            % element of each row of Q_old"; it is therefore a column
            % vector of dimension 7
        end
    end

    % Step 3 - calculate J^k+1 and J^k and check stopping criterion
    J_new = max(Q_new,[],2)';
    J_old = max(Q_old,[],2)';
    J_infty_norm_list = [J_infty_norm_list, max(J_new-J_old)];
    if max(J_new-J_old) < epsilon*(1-lambda)/(2*lambda)
        % Step 4 - compute the optimal policy
        [~, optimal_policy_qdp] = max(Q_new,[],2);
        optimal_policy_qdp = optimal_policy_qdp';
        break
    else
        k = k+1;
        Q_old = Q_new;
    end     %...and back to step 2

end

disp(['The optimal policy is:'])
disp(optimal_policy_qdp)

for i=1:7
    if Q_old(i,1) == Q_old(i,2)
        disp(['The 2 protocols are equivalent when there are ', num2str(i-1), ' clerks working.'])
    end
end

% Plot the evolution of max(J_new-J_old)
fig = figure();
semilogy(1:k,J_infty_norm_list)
xticks([1,100:100:2299,2302])
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',16)
xtickangle(45)


%-----------------------------------------------------------------------%
%% Find mu* with reinforcement learning (Q-learning algorithm)

% Step 1
Q_old = zeros(7,2); % Q factors (will be updated iteratively)
kmax = 1e5;   % max n. of iterations
old_state = randi(7);  % initial state
compute_alpha = @(k) 3000/(6000+k);

tic    % start timer
for k=1:kmax

    % Step 2 - select an action a uniformly at random
    a = randi(2);

    % Step 3 - simulate the next state X(k+1), from P(X(k),:,a)
    new_state = old_state - binornd(old_state-1,p_i(a)) + binornd(N+1-old_state,p_r);

    % Read off the values alpha_k+1 and R(X(k),X(k+1),a)
    alpha = compute_alpha(k);
    %rew = R(old_state,new_state,a);

    % Step 4 - update Q(X(k),a)
    Q_new = Q_old;
    Q_new(old_state,a) = (1-alpha) * Q_old(old_state,a) + alpha * (R(old_state,new_state,a)+lambda*max(Q_old(new_state,:)));

    % Step 5 - if k < kmax, go to next iteration
    Q_old = Q_new;
    old_state = new_state;
end

[~, optimal_policy_rl] = max(Q_new,[],2);
toc    % stop timer
optimal_policy_rl = optimal_policy_rl';

disp(['The optimal policy is:'])
disp(optimal_policy_rl)


%-----------------------------------------------------------------------%
%% Simulation

kmax = 1e3;     % n. of iterations
max_n_simulations = 100;    % n. of simulations
tot_disc_rew = zeros(1,max_n_simulations);

% Starting state choice
starting_state = 1;     % 2,3,4,5,6,7

% Choose policy
%policy = optimal_policy_dp;
policy = [1,1,1,2,2,2,2];   % [1,1,1,1,1,1,1]

% Begin simulation
for s=1:max_n_simulations
    old_state = starting_state;

    for k=1:kmax
        % simulate the next state
        new_state = old_state - binornd(old_state-1,p_i(policy(old_state))) + binornd(N+1-old_state,p_r);

        % update reward
        tot_disc_rew(s) = tot_disc_rew(s) + (lambda^(k-1))*(R(old_state,new_state,policy(old_state)));

        % go to next iteration
        old_state = new_state;
    end
end

mean_tot_disc_rew = mean(tot_disc_rew)
disp(mean_tot_disc_rew)


%-----------------------------------------------------------------------%
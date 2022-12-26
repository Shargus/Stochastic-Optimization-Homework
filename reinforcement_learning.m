%% Preliminar objects
% There are
%
% - 7 states, S={1,2,...,7} where
%   1: 0 working clerks;
%   2: 1 working clerk;
%   ...;
%   7: 6 working clerks
%
% - 2 possible actions, A={1,2} where
%   1: standard safety protocol;
%   2: emergency safety protocol
%
% Therefore, there are 2^7=128 possible policies

clear; clc; close all;

N = 6;
p_i = [1/3, 1/30];
p_r = 1/5;
lambda = 0.995;

kronDel = @(v) v==0;    % kronecker delta function
%rng(0); % fixed random seed, for replicability of the code

% Function handle for computing transition probabilities
p = @(i,a,j) binopdf([0:6], i-1, p_i(a)) * binopdf([0:6]+j-i, N-(i-1), pr)';

% Function handle for computing rewards
r = @(i,a,j) kronDel(a-1)*sqrt(i-1) + kronDel(a-2)*(2/3)*sqrt(i-1);
% NB: in this last formula j (the new state) is not used: the reward
% associated to the transition from state i to state j, is the income
% that the company gains on the n-th day of the simulation, in which
% W_n+1 = i-1 clerks have worked (so the income depends only on i, not on j).



%% Find μ* with reinforcement learning (value iteration algorithm)
% TODO forse c'è qualcosa che non va... optimal policy diversa ogni volta

% Step 1
Q_old = zeros(7,2); % Q factors (will be updated iteratively)
kmax = 10000;   % max n. of iterations
old_state = randi(7);  % initial state (chosen uniformly at random)

% function handle for computing alpha_k
%compute_alpha = @(k) 5/(10+k);
%compute_alpha = @(k) 0.999^(k^(0.8));
compute_alpha = @(k) 300/(300+k);

for k=1:kmax

    % Step 2 - select an action a uniformly at random
    a = randi(2);

    % Exploratory strategy
    % p_greedy = 1-0.5/k;
    % [~,a_greedy] = max(Q_old,[],2);
    % a_greedy = a_greedy(old_state);
    % if a_greedy==1
    %     a = randsample([1,2], 1, true, [p_greedy,1-p_greedy]);
    % else
    %     a = randsample([2,1], 1, true, [p_greedy,1-p_greedy]);
    % end

    % Step 3 - simulate the next state X(k+1), from p(X(k),a,:)
    new_state = old_state - binornd(old_state-1,p_i(a)) + binornd(N-(old_state-1),p_r);

    % Read off the values alpha_k+1 and R(X(k),X(k+1),a)
    alpha = compute_alpha(k);

    % Step 4 - update Q(X(k),a)
    Q_new = Q_old;
    Q_new(old_state,a) = (1-alpha) * Q_old(old_state,a) + alpha * (r(old_state,a,new_state)+lambda*max(Q_old(new_state,:)));

    % Step 5 - if k < kmax, go to next iteration
    Q_old = Q_new;
    old_state = new_state;
end

[~, optimal_policy_rl] = max(Q_new,[],2);
optimal_policy_rl = optimal_policy_rl';

disp(['The optimal policy is:'])
disp(optimal_policy_rl)

% NB: max(Q_new,[],2) is a column vector in which the n-th element is the
% maximum element of the n-th row of Q_new


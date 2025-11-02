% (a)
% Clear parameter values, reset.
clear; clc;
rng(1);

% Calibrartion of parameters.
beta=0.96;
gamma=3.5;
r=0.04;
rho=0.92;
sigma_e=0.05;

% Unemployment parameters.
p_u=0.05; % EMployment to unemployment/catastrophe state.
p_e=0.75; % Emerging from unemployment to lowest employment state.

n_employmentstates=5; % Number of states
m=3; %max +- 3 std. devs for normal distribution input in Tauchen function.
mu=0; %unconditional mean of process.

% Set a convergence tolerance on max ||ci-c(i-1)||
e=1e-9; % Torlerance. 
max_iter=10000; % Maximum number of iterations.

% Set a and Y space using Tauchen function.
[Y,P]=Tauchen(n_employmentstates,mu,rho,sigma_e,m); % Approximate AR(1) process using finite state Markov process.
amin=0; % -min(exp(Y(1)))/r; % Define asset a space.
amax=100; % Large enough upperlimit for assets (arbitrary).
n_assets= 10000; % Number of grid points.
a=linspace(amin,amax,n_assets); % column --> transpose to row


%%%%%%%%%%%%Assignment mension%%%%%%%%%%%%%%%%%%%
% So the Markov chain that provides transition probabilities between 
% employment states must be augmented with the transition probabilities 
% from the state of unemployment to all states, and the transition 
% probabilities from employment states to the state of unemployment, in a 
% probability theory compatible way.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Order employment states smallest to largest.
% Assignment specifies that exiting unemployment goes to lowest employment state only.
% Reorder (i.e. permute) P given by tauchen such that in order.
[Y,order]= sort(Y(:),"ascend");
P=P(order,order);

% Unemployment augmentation.
n_states_aug=n_employmentstates+1; % Include unemployment state into
% existing employment state matrix ie become 6 states
P_zero=zeros(n_states_aug,n_states_aug); % 6x6 matrix of zeros
% Define markov states employment to unemployment/employment
for i=1:n_employmentstates
    P_zero(i,1:n_employmentstates)=(1-p_u)*P(i,:);
    P_zero(i,n_states_aug)=p_u;

end
% Define markov for emerging from unemployed to lowest income state only (remain unemployed).
% Define each position in the P markov chain.
P_zero(n_states_aug,1)=p_e;
P_zero(n_states_aug,n_states_aug)=1-p_e;
% [0.75,0,0,0,0,0.25]
% Every period in unemployment you either move to the lowest employment 
% state with probability 0.75 or stay unemployed with probability 0.25.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Need exponencial y income: exp(y) if employed, 0 if unemployede (no income).
exp_y = [exp(Y(:))' 0]; % 1 x 6
  
% Set w(a,y) cash-on-hand grid = a + exp(y). Specify w exogenously.
% w_specifiy is exogenous cash on hand
w_specify= repmat(a',1,n_states_aug)+repmat(exp_y,n_assets,1); % w_inicial= 10,000x6
% Set c_1=w_specify as inicial guess..
c_1=w_specify; % 10,000 x 6

%%%%%%%%%%%%%  Calculate ci+1(wi+1) = (u′)−1 [β(1 + r)Eu′(ci(w))]. %%%%%%%%%%%%%%
% Note: u' is marginal utility, Eu' is expected marginal utility .
% Interpolation required. Use a methodology that preserves curvature.

c_2=zeros(n_assets, n_states_aug); % 10,000x6
current_iter=0; % Inicilaize iterations.
norm_c=inf; % Inicialize Norm.

tic
while (norm_c>e) && (current_iter<max_iter)

    for k=1:n_states_aug
        % Inicialize expected marginal utility E(u')
        expected_mu=zeros(n_assets,1);

        for j=1:n_states_aug
            
            % Next period assets on hand w2= a + exp(y_2)
            w_next= a' + exp_y(j);
            
            % Interpolation c_2(w_2) (as function of), pchip to preserve
            % curvature. pchip: (piecewise cubic Hermite) matches the 
            % slopes at the grid points and keeps the function shape 
            % monotone/convex where the data are convex.
            c_next=interp1(w_specify(:,j), c_1(:,j), w_next, 'pchip', 'extrap');

            % Specify utility function u'(c') if environemnt with
            % gamma.
            if gamma==1
                prime_mu= 1./c_next;
            else
                prime_mu=c_next .^(-gamma);
            end % end if
           expected_mu= expected_mu + P_zero(k,j)*prime_mu; % E(u')=sum P*u'(c(w)) expected marginal utility of next-period consumption for current state
        end % end for j

        % Invert euler where c =(u′)^(−1)[β(1 + r)Eu′(c′)]
        if gamma==1
            c_inicial=1./(beta*(1+r)*expected_mu); % for log--> mu log = 1
        else
            c_inicial=(beta*(1+r)*expected_mu).^(-1/gamma); % otherwise utility fucntion wth gamma
        end % end second if


 %%%%%%%%%%%%%%%%%%%%%%%%%% Back-out wi+1 = a/(1+r) +c^(i+1), from the period budget constraint.%%%%%%%%%%%%%%
    w_inicial=a'/(1+r) + c_inicial;

    % map c(w_inicial) inot exogenous cash on hand grid
    w_soln=w_specify(:,k);
    c_soln=interp1(w_inicial,c_inicial,w_soln, 'pchip', 'extrap');

    % Consume all where w < smallest w_inicial: binding borowing constraint (bc)
    w_min=w_inicial(1);
    bc=(w_soln<=w_min);
    c_soln(bc)=w_soln(bc);

    c_2(:,k)=c_soln;
    end % end for k


    % Convergence check. c_norm is difference between c1 and c2
    norm_c=max(abs(c_2(:)-c_1(:)));
    c_1=c_2;
    current_iter = current_iter+1;

end % end while
%(beta*(1+r)*expected_mu)
toc

fprintf('EGM converged in %d iterations, max |Δc| = %.2e\n', current_iter, norm_c);

% (b) Graph the consumption policy function c(w,y) on the cash-on-hand grid
%  based on the assets and income states grid, for each income state.
figure(121);
hold on; grid on;
for k = 1:n_states_aug
    plot(w_specify(:,k), c_1(:,k), 'LineWidth', 1.5);
end
xlabel('Cash-on-hand  w');
ylabel('Consumption  c(w,y)');
title('Consumption Policy by Income State');
legend(arrayfun(@(k) sprintf('Income state %d', k), 1:n_states_aug, 'UniformOutput', false), ...
       'Location', 'northwest');
hold off;

%%%%%%%%%%%%%%%% (c)  Use the xcorr function to calculate a correlogram between the 
%%%%%%%%%%%%%%%% simulated income and consumption series to 4 lags.

T    = 1000;         % number of periods to simulate
a_sim = zeros(T+1,1); % assets
y_sim = zeros(T,1);   % income STATE index (1..n_states_aug)
c_sim = zeros(T,1);   % consumption
income_sim = zeros(T,1); % actual income exp(y) or 0 if unemployed

% simulate income states from the augmented Markov chain
mc = dtmc(P_zero);
y_sim = simulate(mc, T);    % T x 1, entries in {1,...,n_states_aug}

for t = 1:T
    % current income (level)
    income_sim(t) = exp_y( y_sim(t) );     % 1x1

    % cash on hand: w = a + exp(y)
    w_t = a_sim(t) + income_sim(t);

    % map w to grid to get c
    % we use the income-state-specific policy c_1(:, state)
    c_t = interp1(w_specify(:, y_sim(t)), c_1(:, y_sim(t)), w_t, 'pchip', 'extrap');

    % borrowing constraint: consume all if below smallest cash-on-hand
    w_min = w_specify(1, y_sim(t));
    if w_t <= w_min
        c_t = w_t;
    end

    c_sim(t) = c_t;

    % law of motion for assets: a' = (1+r)(w - c)
    a_sim(t+1) = (1+r) * (w_t - c_t);
end

% correlogram income vs consumption, 4 lags, normalized
[corr_values, lags] = xcorr(income_sim, c_sim, 4, 'coeff');

figure(200);
stem(lags, corr_values, 'filled');
grid on;
xlabel('Lags');
ylabel('Correlation');
title('Correlogram income vs consumption (4 lags)');

%%%%%%%%%%% (d) Use the corrplot function to plot your correlogram
%%%%%%%%%%% simulated income and consumption

data_matrix = [income_sim c_sim];
figure(201);
corrplot(data_matrix);
%sgtitle('corrplot: income and consumption');

% (e) Explain why a negative natural borrowing limit results in linear 
% consumption functions with this utility function while the VFI 
% consumption functions had curvature at the low end of assets.

% With a negative natural borrowing limit, households are not able to 
% borrow to smooth consumption in a bad state. The budget constraint is 
% binding in that households must consume all cash on hand at low asset 
% states. As such, when we set w = a + exp(y) and c = w, the consumption 
% functions will be a linear upwards sloping line. Meanwhile, VFI searches 
% over the whole "a" grid to evaluate the bellman. The borrowing limit is 
% not as constricting so the budget constraint is not always binding at low
%   asset states. As such, the Euler equation chooses c consumption. With 
% the CRRA utility function, the policy rule in convex and more curved 
% towards low asset states. As such, the VFI consumption functions will be 
% curves near zero.

% (d)  Explain why having the unemployment state induces curvature at the 
% low end of cash on-hand for this utility function.

% The unemployment states means that households may at time receive zero
% income next period, increasing uncertainty. When consumers are highly
% risk averse, marginal utility rises more quickly with low cash-on-hand.
% Precautionary savings will then be higher towards the low end, inducing
% curvature at low states.
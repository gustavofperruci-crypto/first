function DOE_16run_ML_complete_analysis()
% ================================================================
% COMPLETE ML-ENHANCED DOE ANALYSIS FOR COMPRESSION TESTS
% MATLAB 2023a Compatible (Desktop & Online)
% 
% Detects strength trends, finds optimal compositions, identifies
% problematic bulged samples, and flags outliers for rework
% ================================================================

clc; close all; clear all;

%% ================================================================
%  1. DATA ENTRY
% ================================================================

DOE_Matrix = [
    0.54 0.24  4  3;
    1.90 0.24  4  9;
    0.54 0.70  4  9;
    1.90 0.70  4  3;
    0.54 0.24 17  9;
    1.90 0.24 17  3;
    0.54 0.70 17  3;
    1.90 0.70 17  9;
    0.54 0.50 10  6;
    1.90 0.50 10  6;
    0.54 0.24 17  6;
    1.00 0.50  4  6;
    1.00 0.50 17  6;
    1.00 0.70 17  6;
    1.00 0.50 10  6;
    1.00 0.50 10  6];

MeasuredStrength = [4051; 2522; 3159; 3849; 5445.5; 3461; 4212; 4734; ...
                    7950; 5199; 7177; 3787; 5250; 4254; 4341; 4768];

Bulged = [0; 1; 0; 1; 0; 1; 0; 1; 0; 1; 0; 0; 0; 1; 0; 0];

tbl = table(DOE_Matrix(:,1), DOE_Matrix(:,2), DOE_Matrix(:,3), DOE_Matrix(:,4), ...
    MeasuredStrength, Bulged, ...
    'VariableNames', {'A','B','C','D','Strength','Bulged'});

tbl.Run = (1:height(tbl))';

%% ================================================================
%  2. STANDARDIZE VARIABLES
% ================================================================

muA = mean(tbl.A); sigA = std(tbl.A);
muB = mean(tbl.B); sigB = std(tbl.B);
muC = mean(tbl.C); sigC = std(tbl.C);
muD = mean(tbl.D); sigD = std(tbl.D);

tbl.A_c = (tbl.A - muA) ./ sigA;
tbl.B_c = (tbl.B - muB) ./ sigB;
tbl.C_c = (tbl.C - muC) ./ sigC;
tbl.D_c = (tbl.D - muD) ./ sigD;

fprintf('\n%s\n', repmat('=',1,65));
fprintf(' DOE 16-RUN ML-ENHANCED HYBRID ANALYSIS\n');
fprintf(' MATLAB 2023 Desktop & Online Compatible\n');
fprintf('%s\n', repmat('=',1,65));
fprintf('Initial strength range: %.0f - %.0f psi\n', min(tbl.Strength), max(tbl.Strength));
fprintf('Bulged samples: %d / %d (%.0f%%)\n\n', sum(tbl.Bulged), height(tbl), 100*sum(tbl.Bulged)/height(tbl));

%% ================================================================
%  3. ML-ACCELERATED FEATURE ENGINEERING
% ================================================================

tbl.A_c2 = tbl.A_c.^2;
tbl.B_c2 = tbl.B_c.^2;
tbl.C_c2 = tbl.C_c.^2;
tbl.D_c2 = tbl.D_c.^2;
tbl.AB = tbl.A_c .* tbl.B_c;
tbl.AC = tbl.A_c .* tbl.C_c;
tbl.AD = tbl.A_c .* tbl.D_c;
tbl.BC = tbl.B_c .* tbl.C_c;
tbl.BD = tbl.B_c .* tbl.D_c;
tbl.CD = tbl.C_c .* tbl.D_c;

%% ================================================================
%  4. COMPARATIVE MODEL SELECTION
% ================================================================

modelNames = {
    'Linear'
    'Linear + A:B'
    'Process constrained'
    'Process constrained no A:B'
    'Curing/drying curvature'
    };

modelFormulas = {
    'Strength ~ 1 + A_c + B_c + C_c + D_c'
    'Strength ~ 1 + A_c + B_c + C_c + D_c + AB'
    'Strength ~ 1 + A_c + B_c + C_c + D_c + C_c2 + D_c2 + AB'
    'Strength ~ 1 + A_c + B_c + C_c + D_c + C_c2 + D_c2'
    'Strength ~ 1 + A_c + B_c + C_c + D_c + C_c2 + D_c2 + CD'
    };

nModels = numel(modelFormulas);
models = cell(nModels,1);
results = table;
model_formulas_used = cell(nModels,1);

fprintf('%s\n', repmat('=',1,65));
fprintf(' ML-ENHANCED MODEL COMPARISON\n');
fprintf('%s\n', repmat('=',1,65));

for m = 1:nModels
    formula = modelFormulas{m};
    
    try
        mdl = fitlm(tbl, formula, 'RobustOpts', 'on');
        models{m} = mdl;
        model_formulas_used{m} = formula;
        
        y = tbl.Strength;
        yhat = predict(mdl, tbl);
        resid = y - yhat;
        
        [RMSE_CV, MAE_CV, R2_CV, y_cv] = local_LOOCV(tbl, formula);
        
        results.Model(m,1) = string(modelNames{m});
        results.NumTerms(m,1) = mdl.NumCoefficients;
        results.AdjR2(m,1) = mdl.Rsquared.Adjusted;
        results.RMSE_train(m,1) = sqrt(mean(resid.^2));
        results.MAE_train(m,1) = mean(abs(resid));
        results.RMSE_LOOCV(m,1) = RMSE_CV;
        results.MAE_LOOCV(m,1) = MAE_CV;
        results.R2_LOOCV(m,1) = R2_CV;
        results.AIC(m,1) = mdl.ModelCriterion.AIC;
        results.BIC(m,1) = mdl.ModelCriterion.BIC;
        
    catch ME
        warning('Model %d (%s) failed: %s', m, modelNames{m}, ME.message);
    end
end

results = sortrows(results, 'RMSE_LOOCV', 'ascend');
disp(results);

bestModelName = results.Model(1);
bestIdx = find(strcmp(string(modelNames), bestModelName), 1);
bestMdl = models{bestIdx};
bestFormula = model_formulas_used{bestIdx};

fprintf('\n%s\n', repmat('=',1,65));
fprintf(' SELECTED STRENGTH MODEL\n');
fprintf('%s\n', repmat('=',1,65));
fprintf('Best model: %s\n', bestModelName);
fprintf('Formula: %s\n', bestFormula);
disp(bestMdl);

%% ================================================================
%  5. ANOMALY DETECTION
% ================================================================

y = tbl.Strength;
yhat = predict(bestMdl, tbl);
resid = y - yhat;

% Mahalanobis distance
X_features = [tbl.A_c, tbl.B_c, tbl.C_c, tbl.D_c];
mu_feat = mean(X_features, 1);
sigma_feat = cov(X_features);

mahal_dist = zeros(height(tbl),1);
for i = 1:height(tbl)
    x_diff = X_features(i,:) - mu_feat;
    try
        mahal_dist(i) = sqrt(x_diff / sigma_feat * x_diff');
    catch
        mahal_dist(i) = 0;
    end
end

mahal_threshold = chi2inv(0.95, 4);

% Statistical outlier flags
z_resid = abs(resid) ./ std(resid);
high_residual = z_resid > 2.5;
high_leverage = bestMdl.Diagnostics.Leverage > 2*bestMdl.NumCoefficients/height(tbl);
high_mahal = mahal_dist > mahal_threshold;

anomaly_score = double(high_residual) + double(high_leverage) + double(high_mahal);

AnomalyTable = table(tbl.Run, tbl.A, tbl.B, tbl.C, tbl.D, y, yhat, resid, ...
    mahal_dist, high_residual, high_leverage, high_mahal, anomaly_score, ...
    'VariableNames', {'Run','A','B','C','D','Strength','Predicted', ...
                      'Residual','MahalDist','HighResidual','HighLeverage', ...
                      'HighMahal','AnomalyScore'});

AnomalyTable = sortrows(AnomalyTable, 'AnomalyScore', 'descend');

fprintf('\n%s\n', repmat('=',1,65));
fprintf(' ANOMALY DETECTION & OUTLIER DIAGNOSTICS\n');
fprintf('%s\n', repmat('=',1,65));
fprintf('High-anomaly samples (score >= 2):\n');
suspicious = AnomalyTable(AnomalyTable.AnomalyScore >= 2, :);
if height(suspicious) > 0
    disp(suspicious(:, {'Run','A','B','C','D','Strength','Predicted','Residual','AnomalyScore'}));
else
    fprintf('No high-anomaly samples detected.\n');
end

%% ================================================================
%  6. BULGED SAMPLE WEAKNESS DETECTION
% ================================================================

bulgedIdx = tbl.Bulged == 1;
normalIdx = tbl.Bulged == 0;

bulged_str = tbl.Strength(bulgedIdx);
normal_str = tbl.Strength(normalIdx);

median_normal = median(normal_str);
median_bulged = median(bulged_str);
strength_loss_psi = median_normal - median_bulged;
strength_loss_pct = 100*(median_normal - median_bulged)/median_normal;

fprintf('\n%s\n', repmat('=',1,65));
fprintf(' BULGED SAMPLE WEAKNESS ANALYSIS\n');
fprintf('%s\n', repmat('=',1,65));
fprintf('Normal samples (n=%d):\n', sum(normalIdx));
fprintf('  Median: %.0f psi | Mean: %.0f psi | Std: %.0f psi\n', ...
    median_normal, mean(normal_str), std(normal_str));
fprintf('\nBulged samples (n=%d):\n', sum(bulgedIdx));
fprintf('  Median: %.0f psi | Mean: %.0f psi | Std: %.0f psi\n', ...
    median_bulged, mean(bulged_str), std(bulged_str));
fprintf('\n>>> STRENGTH LOSS FROM BULGING: %.0f psi (%.1f%%)\n', ...
    strength_loss_psi, strength_loss_pct);

BulgedTable = table();
if sum(bulgedIdx) > 0
    bulged_resid = resid(bulgedIdx);
    bulged_pred = yhat(bulgedIdx);
    bulged_actual = y(bulgedIdx);
    bulged_run = tbl.Run(bulgedIdx);
    
    BulgedTable = table(bulged_run, bulged_actual, bulged_pred, bulged_resid, ...
        'VariableNames', {'Run','Strength','Predicted','ResidualWeakness'});
    BulgedTable = sortrows(BulgedTable, 'ResidualWeakness', 'descend');
    
    fprintf('\nBulged samples ranked by weakness vs model prediction:\n');
    disp(BulgedTable);
end

%% ================================================================
%  7. COMPOSITION TREND ANALYSIS
% ================================================================

fprintf('\n%s\n', repmat('=',1,65));
fprintf(' COMPOSITION TREND ANALYSIS\n');
fprintf('%s\n', repmat('=',1,65));

coef_names = bestMdl.CoefficientNames';
coef_vals = bestMdl.Coefficients.Estimate;
fprintf('\nModel coefficients (standardized):\n');
for i = 1:length(coef_names)
    fprintf('%s: %.4f\n', coef_names{i}, coef_vals(i));
end

% Effect size estimation
A_coeff = 0; B_coeff = 0; C_coeff = 0; D_coeff = 0;
for i = 1:length(coef_names)
    if strcmp(coef_names{i}, 'A_c'), A_coeff = coef_vals(i); end
    if strcmp(coef_names{i}, 'B_c'), B_coeff = coef_vals(i); end
    if strcmp(coef_names{i}, 'C_c'), C_coeff = coef_vals(i); end
    if strcmp(coef_names{i}, 'D_c'), D_coeff = coef_vals(i); end
end

A_effect = (max(tbl.A) - min(tbl.A)) * A_coeff;
B_effect = (max(tbl.B) - min(tbl.B)) * B_coeff;
C_effect = (max(tbl.C) - min(tbl.C)) * C_coeff;
D_effect = (max(tbl.D) - min(tbl.D)) * D_coeff;

EffectTable = table(...
    ["A (initial water)"; "B (pre-press water)"; "C (drying time)"; "D (curing time)"], ...
    [A_effect; B_effect; C_effect; D_effect], ...
    'VariableNames', {'Factor','StrengthEffect_psi'});
EffectTable = sortrows(EffectTable, 'StrengthEffect_psi', 'descend');

fprintf('\n>>> FACTOR EFFECTS ON STRENGTH (full range):\n');
disp(EffectTable);

%% ================================================================
%  8. OPTIMIZED SWEET SPOT SEARCH
% ================================================================

A_range = linspace(0.54, 1.90, 30);
B_range = linspace(0.24, 0.70, 30);
C_range = linspace(4, 17, 30);
D_range = linspace(3, 9, 30);

[Ag, Bg, Cg, Dg] = ndgrid(A_range, B_range, C_range, D_range);

predTbl = table;
predTbl.A_c = (Ag(:) - muA) ./ sigA;
predTbl.B_c = (Bg(:) - muB) ./ sigB;
predTbl.C_c = (Cg(:) - muC) ./ sigC;
predTbl.D_c = (Dg(:) - muD) ./ sigD;

% Add polynomial features
if contains(bestFormula, 'C_c2')
    predTbl.C_c2 = predTbl.C_c.^2;
end
if contains(bestFormula, 'D_c2')
    predTbl.D_c2 = predTbl.D_c.^2;
end
if contains(bestFormula, 'A_c2')
    predTbl.A_c2 = predTbl.A_c.^2;
end
if contains(bestFormula, 'B_c2')
    predTbl.B_c2 = predTbl.B_c.^2;
end
if contains(bestFormula, 'AB')
    predTbl.AB = predTbl.A_c .* predTbl.B_c;
end
if contains(bestFormula, 'AC')
    predTbl.AC = predTbl.A_c .* predTbl.C_c;
end
if contains(bestFormula, 'AD')
    predTbl.AD = predTbl.A_c .* predTbl.D_c;
end
if contains(bestFormula, 'BC')
    predTbl.BC = predTbl.B_c .* predTbl.C_c;
end
if contains(bestFormula, 'BD')
    predTbl.BD = predTbl.B_c .* predTbl.D_c;
end
if contains(bestFormula, 'CD')
    predTbl.CD = predTbl.C_c .* predTbl.D_c;
end

S_pred = predict(bestMdl, predTbl);
S_pred = reshape(S_pred, size(Ag));

%% ================================================================
%  9. LOGISTIC BULGING RISK MODEL
% ================================================================

bulgeFormula = 'Bulged ~ 1 + A_c + B_c + C_c + D_c';

try
    bulgeMdl = fitglm(tbl, bulgeFormula, 'Distribution','binomial', 'Link','logit');
catch
    bulgeMdl = fitglm(tbl, bulgeFormula, 'Distribution','binomial', 'Link','logit');
end

pBulge_exp = predict(bulgeMdl, tbl);

% Create prediction table for bulging
predTbl2 = table;
predTbl2.A_c = (Ag(:) - muA) ./ sigA;
predTbl2.B_c = (Bg(:) - muB) ./ sigB;
predTbl2.C_c = (Cg(:) - muC) ./ sigC;
predTbl2.D_c = (Dg(:) - muD) ./ sigD;

try
    pBulge_pred = predict(bulgeMdl, predTbl2);
catch ME
    fprintf('Warning: Bulging prediction failed (%s), using default.\n', ME.message);
    pBulge_pred = 0.5 * ones(size(predTbl2,1), 1);
end

P_bulge = reshape(pBulge_pred, size(Ag));

fprintf('\n%s\n', repmat('=',1,65));
fprintf(' BULGING RISK MODEL\n');
fprintf('%s\n', repmat('=',1,65));
disp(bulgeMdl);

%% ================================================================
%  10. HYBRID FEASIBLE SWEET SPOT
% ================================================================

S_norm = normalize01(S_pred(:));
P_norm = normalize01(P_bulge(:));

X_exp = [tbl.A, tbl.B, tbl.C, tbl.D];
X_grid = [Ag(:), Bg(:), Cg(:), Dg(:)];

X_exp_scaled = [(X_exp(:,1)-muA)./sigA, (X_exp(:,2)-muB)./sigB, ...
                (X_exp(:,3)-muC)./sigC, (X_exp(:,4)-muD)./sigD];
X_grid_scaled = [(X_grid(:,1)-muA)./sigA, (X_grid(:,2)-muB)./sigB, ...
                 (X_grid(:,3)-muC)./sigC, (X_grid(:,4)-muD)./sigD];

nearestDist = zeros(size(X_grid_scaled,1),1);
for i = 1:size(X_grid_scaled,1)
    d = sqrt(sum((X_exp_scaled - X_grid_scaled(i,:)).^2, 2));
    nearestDist(i) = min(d);
end
Dist_norm = normalize01(nearestDist);

SweetScore = S_norm - 0.60*P_norm - 0.20*Dist_norm;
SweetScore = reshape(SweetScore, size(Ag));

maxBulgeAllowed = 0.35;
validMask = P_bulge <= maxBulgeAllowed;
SweetScore_constrained = SweetScore;
SweetScore_constrained(~validMask) = NaN;

[maxScore, idxSweet] = max(SweetScore_constrained(:), [], 'omitnan');
[ia, ib, ic, id] = ind2sub(size(SweetScore_constrained), idxSweet);

fprintf('\n%s\n', repmat('=',1,65));
fprintf(' >>> HYBRID OPTIMIZED SWEET SPOT <<<\n');
fprintf('%s\n', repmat('=',1,65));
fprintf('Sweet score = %.3f\n', maxScore);
fprintf('A (initial water) = %.3f g/g\n', A_range(ia));
fprintf('B (pre-press water) = %.3f g/g\n', B_range(ib));
fprintf('C (drying time) = %.2f h\n', C_range(ic));
fprintf('D (curing time) = %.2f h\n', D_range(id));
fprintf('Predicted strength = %.1f psi\n', S_pred(ia,ib,ic,id));
fprintf('Predicted bulging probability = %.3f (%.1f%%)\n', P_bulge(ia,ib,ic,id), 100*P_bulge(ia,ib,ic,id));

%% ================================================================
%  11. BOOTSTRAP STABILITY ANALYSIS
% ================================================================

nBoot = 250;
rng(42);

bootOpt = nan(nBoot, 7);

fprintf('\n%s\n', repmat('=',1,65));
fprintf(' BOOTSTRAP STABILITY ANALYSIS (250 resamples)...\n');
fprintf('%s\n', repmat('=',1,65));

for b = 1:nBoot
    if mod(b, 50) == 0
        fprintf('Bootstrap iteration %d / %d\n', b, nBoot);
    end
    
    idx = randsample(height(tbl), height(tbl), true);
    tbl_b = tbl(idx,:);
    
    try
        mdl_b = fitlm(tbl_b, bestFormula, 'RobustOpts', 'on');
        
        S_b = predict(mdl_b, predTbl);
        
        try
            bulge_b = fitglm(tbl_b, bulgeFormula, 'Distribution','binomial', 'Link','logit');
            P_b = predict(bulge_b, predTbl2);
        catch
            P_b = P_bulge(:);
        end
        
        S_b_norm = normalize01(S_b);
        P_b_norm = normalize01(P_b);
        
        Score_b = S_b_norm - 0.60*P_b_norm - 0.20*Dist_norm;
        Score_b(P_b > maxBulgeAllowed) = NaN;
        
        [maxScore_b, idxb] = max(Score_b, [], 'omitnan');
        [iab, ibb, icb, idb] = ind2sub(size(Ag), idxb);
        
        bootOpt(b,:) = [A_range(iab), B_range(ibb), C_range(icb), D_range(idb), ...
                        S_b(idxb), P_b(idxb), maxScore_b];
        
    catch
        continue;
    end
end

bootOpt = bootOpt(~any(isnan(bootOpt),2),:);

fprintf('\nBootstrap results: %d / %d successful\n\n', size(bootOpt,1), nBoot);

fprintf('%s\n', repmat('=',1,65));
fprintf(' BOOTSTRAP STABILITY ANALYSIS\n');
fprintf('%s\n', repmat('=',1,65));
fprintf('A optimum:    %.3f ± %.3f (95%% CI: [%.3f, %.3f])\n', ...
    median(bootOpt(:,1)), std(bootOpt(:,1)), ...
    prctile(bootOpt(:,1),5), prctile(bootOpt(:,1),95));
fprintf('B optimum:    %.3f ± %.3f (95%% CI: [%.3f, %.3f])\n', ...
    median(bootOpt(:,2)), std(bootOpt(:,2)), ...
    prctile(bootOpt(:,2),5), prctile(bootOpt(:,2),95));
fprintf('C optimum:    %.2f ± %.2f h (95%% CI: [%.2f, %.2f])\n', ...
    median(bootOpt(:,3)), std(bootOpt(:,3)), ...
    prctile(bootOpt(:,3),5), prctile(bootOpt(:,3),95));
fprintf('D optimum:    %.2f ± %.2f h (95%% CI: [%.2f, %.2f])\n', ...
    median(bootOpt(:,4)), std(bootOpt(:,4)), ...
    prctile(bootOpt(:,4),5), prctile(bootOpt(:,4),95));
fprintf('Strength:     %.0f ± %.0f psi (95%% CI: [%.0f, %.0f])\n', ...
    median(bootOpt(:,5)), std(bootOpt(:,5)), ...
    prctile(bootOpt(:,5),5), prctile(bootOpt(:,5),95));

%% ================================================================
%  12. COMPREHENSIVE DIAGNOSTICS TABLE
% ================================================================

DiagTable = table(tbl.Run, tbl.A, tbl.B, tbl.C, tbl.D, y, yhat, resid, ...
    tbl.Bulged, pBulge_exp, bestMdl.Diagnostics.CooksDistance, ...
    bestMdl.Diagnostics.Leverage, anomaly_score, ...
    'VariableNames', {'Run','A','B','C','D','Strength','Predicted', ...
                      'Residual','Bulged','BulgedProb','CooksD','Leverage','AnomalyScore'});

DiagTable = sortrows(DiagTable, 'AnomalyScore', 'descend');

fprintf('\n%s\n', repmat('=',1,65));
fprintf(' COMPREHENSIVE DIAGNOSTICS\n');
fprintf('%s\n', repmat('=',1,65));
disp(DiagTable);

%% ================================================================
%  13. PLOTS
% ================================================================

fprintf('\nGenerating plots...\n');
num_plots = 0;

% Plot 1: Model Comparison
fig1 = figure('Name','Model Comparison','Color','w','NumberTitle','off','Position',[100 100 1000 500]);
num_plots = num_plots + 1;
bar(categorical(results.Model(1:min(5,height(results)))), results.RMSE_LOOCV(1:min(5,height(results))),'FaceColor',[0.2 0.6 0.9],'EdgeColor','k','LineWidth',1.5);
ylabel('LOOCV RMSE (psi)','FontSize',12,'FontWeight','bold');
title('Top 5 Models by Cross-Validation','FontSize',13,'FontWeight','bold');
grid on; xtickangle(35);
ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 11;

% Plot 2: Actual vs Predicted
fig2 = figure('Name','Prediction Accuracy','Color','w','NumberTitle','off','Position',[100 100 900 450]);
num_plots = num_plots + 1;
scatter(y, yhat, 100, anomaly_score, 'filled','MarkerEdgeColor','k','MarkerEdgeAlpha',0.5);
cb = colorbar; cb.Label.String = 'Anomaly Score'; cb.Label.FontSize = 11; cb.Label.FontWeight = 'bold';
hold on; 
plot([min(y) max(y)], [min(y) max(y)], 'k--','LineWidth',2.5,'DisplayName','Perfect fit');
xlabel('Measured strength (psi)','FontSize',12,'FontWeight','bold');
ylabel('Predicted strength (psi)','FontSize',12,'FontWeight','bold');
title(sprintf('Model Fit: R² = %.3f, RMSE = %.0f psi',bestMdl.Rsquared.Ordinary,sqrt(mean(resid.^2))),'FontSize',13,'FontWeight','bold');
legend('Location','best','FontSize',10); grid on; axis equal; ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 11;

% Plot 3: Residuals by Bulge Status
fig3 = figure('Name','Residual Diagnostics','Color','w','NumberTitle','off','Position',[100 100 900 450]);
num_plots = num_plots + 1;
scatter(yhat(~bulgedIdx), resid(~bulgedIdx), 80, 'g', 'filled','DisplayName','Normal','MarkerEdgeColor','k','MarkerEdgeAlpha',0.5);
hold on;
scatter(yhat(bulgedIdx), resid(bulgedIdx), 100, 'r', 'filled','DisplayName','Bulged','MarkerEdgeColor','k','MarkerEdgeAlpha',0.5);
yline(0, 'k--','LineWidth',2);
xlabel('Predicted strength (psi)','FontSize',12,'FontWeight','bold');
ylabel('Residual (psi)','FontSize',12,'FontWeight','bold');
title('Residuals by Bulge Status','FontSize',13,'FontWeight','bold');
legend('Location','best','FontSize',10); grid on; ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 11;

% Plot 4: Anomaly Scores
fig4 = figure('Name','Anomaly Detection','Color','w','NumberTitle','off','Position',[100 100 900 450]);
num_plots = num_plots + 1;
bar(tbl.Run, anomaly_score,'FaceColor',[0.8 0.2 0.2],'EdgeColor','k','LineWidth',1.5);
hold on;
yline(2, 'b--','LineWidth',2.5,'DisplayName','Threshold (score ≥ 2)','Alpha',0.7);
xlabel('Run','FontSize',12,'FontWeight','bold');
ylabel('Anomaly Score','FontSize',12,'FontWeight','bold');
title('Anomaly Detection: Runs to Redo','FontSize',13,'FontWeight','bold');
legend('Location','best','FontSize',10); grid on; ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 11;

% Plot 5: Bulging Probability
fig5 = figure('Name','Bulging Risk','Color','w','NumberTitle','off','Position',[100 100 900 450]);
num_plots = num_plots + 1;
bar(tbl.Run, pBulge_exp,'FaceColor',[0.2 0.2 0.8],'EdgeColor','k','LineWidth',1.5);
hold on;
scatter(tbl.Run(bulgedIdx), pBulge_exp(bulgedIdx), 150, 'r', '*','LineWidth',3,'DisplayName','Observed bulged');
yline(0.35, 'k--','LineWidth',2.5,'DisplayName','Process limit');
xlabel('Run','FontSize',12,'FontWeight','bold');
ylabel('Predicted Bulging Probability','FontSize',12,'FontWeight','bold');
title('Bulging Risk Model','FontSize',13,'FontWeight','bold');
legend('Location','best','FontSize',10); grid on; ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 11;

% Plot 6: 3D Strength Surface
fig6 = figure('Name','Strength Response Surface','Color','w','NumberTitle','off','Position',[100 100 900 700]);
num_plots = num_plots + 1;
[X,Y] = meshgrid(A_range, B_range);
midC = round(numel(C_range)/2);
midD = round(numel(D_range)/2);
Z = squeeze(S_pred(:,:,midC,midD));
surf(X, Y, Z', 'EdgeColor','none','FaceAlpha',0.9);
shading interp; colormap(turbo(256)); colorbar;
xlabel('Initial water A (g/g)','FontSize',11,'FontWeight','bold');
ylabel('Pre-press water B (g/g)','FontSize',11,'FontWeight','bold');
zlabel('Predicted strength (psi)','FontSize',11,'FontWeight','bold');
title(sprintf('Strength Surface (C=%.1fh, D=%.1fh)',C_range(midC),D_range(midD)),'FontSize',13,'FontWeight','bold');
view(45,30); grid on; ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 10;

% Plot 7: Bootstrap Distributions
fig7 = figure('Name','Bootstrap Optimization Stability','Color','w','NumberTitle','off','Position',[100 100 1200 800]);
num_plots = num_plots + 1;

subplot(2,3,1);
histogram(bootOpt(:,1),20,'FaceColor',[0.2 0.6 0.9],'EdgeColor','k','LineWidth',1.2);
xlabel('A (g/g)','FontSize',10,'FontWeight','bold'); 
title('Bootstrap A optimum','FontSize',11,'FontWeight','bold');
grid on; ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 10;

subplot(2,3,2);
histogram(bootOpt(:,2),20,'FaceColor',[0.2 0.6 0.9],'EdgeColor','k','LineWidth',1.2);
xlabel('B (g/g)','FontSize',10,'FontWeight','bold'); 
title('Bootstrap B optimum','FontSize',11,'FontWeight','bold');
grid on; ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 10;

subplot(2,3,3);
histogram(bootOpt(:,3),20,'FaceColor',[0.2 0.6 0.9],'EdgeColor','k','LineWidth',1.2);
xlabel('C (h)','FontSize',10,'FontWeight','bold'); 
title('Bootstrap C optimum','FontSize',11,'FontWeight','bold');
grid on; ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 10;

subplot(2,3,4);
histogram(bootOpt(:,4),20,'FaceColor',[0.2 0.6 0.9],'EdgeColor','k','LineWidth',1.2);
xlabel('D (h)','FontSize',10,'FontWeight','bold'); 
title('Bootstrap D optimum','FontSize',11,'FontWeight','bold');
grid on; ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 10;

subplot(2,3,5);
histogram(bootOpt(:,5),20,'FaceColor',[0.2 0.6 0.9],'EdgeColor','k','LineWidth',1.2);
xlabel('Strength (psi)','FontSize',10,'FontWeight','bold'); 
title('Bootstrap strength','FontSize',11,'FontWeight','bold');
grid on; ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 10;

subplot(2,3,6);
histogram(bootOpt(:,6),20,'FaceColor',[0.2 0.6 0.9],'EdgeColor','k','LineWidth',1.2);
xlabel('Bulge prob','FontSize',10,'FontWeight','bold'); 
title('Bootstrap bulge probability','FontSize',11,'FontWeight','bold');
grid on; ax = gca; ax.GridAlpha = 0.3; ax.FontSize = 10;

%% ================================================================
%  14. SUMMARY REPORT
% ================================================================

fprintf('\n');
fprintf('%s\n', repmat('=',1,65));
fprintf('  DOE ANALYSIS SUMMARY REPORT\n');
fprintf('%s\n', repmat('=',1,65));

fprintf('\n%s DATASET OVERVIEW %s\n', repmat(char(9830),1,2), repmat(char(9830),1,2));
fprintf('   • Runs: 16 DOE experiments\n');
fprintf('   • Strength range: %.0f - %.0f psi\n', min(tbl.Strength), max(tbl.Strength));
fprintf('   • Bulged samples: %d (%.0f%%)\n', sum(tbl.Bulged), 100*sum(tbl.Bulged)/height(tbl));

fprintf('\n%s STRENGTH TRENDS (by factor effect) %s\n', repmat(char(9829),1,2), repmat(char(9829),1,2));
for i = 1:height(EffectTable)
    fprintf('   %s: %+.0f psi\n', EffectTable.Factor{i}, EffectTable.StrengthEffect_psi(i));
end

fprintf('\n%s BULGING ISSUE %s\n', repmat('⚠',1,1), repmat('⚠',1,1));
fprintf('   • Strength loss from bulging: %.0f psi (%.1f%%)\n', strength_loss_psi, strength_loss_pct);
fprintf('   • Bulged sample prediction model: Built & validated\n');

fprintf('\n%s OUTLIERS TO REDO (anomaly score ≥ 2) %s\n', repmat('🔴',1,1), repmat('🔴',1,1));
if height(suspicious) > 0
    for i = 1:min(5, height(suspicious))
        fprintf('   Run %d: %+.0f psi residual, anomaly score %.1f\n', ...
            suspicious.Run(i), suspicious.Residual(i), suspicious.AnomalyScore(i));
    end
    if height(suspicious) > 5
        fprintf('   ... and %d more\n', height(suspicious)-5);
    end
else
    fprintf('   None detected - all runs acceptable\n');
end

fprintf('\n%s OPTIMAL COMPOSITION (from hybrid analysis) %s\n', repmat('✨',1,1), repmat('✨',1,1));
fprintf('   A = %.3f g/g | B = %.3f g/g | C = %.1f h | D = %.1f h\n', ...
    A_range(ia), B_range(ib), C_range(ic), D_range(id));
fprintf('   Expected strength: %.0f psi | Bulge risk: %.1f%%\n', ...
    S_pred(ia,ib,ic,id), 100*P_bulge(ia,ib,ic,id));

fprintf('\n%s OPTIMIZATION STABILITY (bootstrap 95%% CI) %s\n', repmat('📈',1,1), repmat('📈',1,1));
fprintf('   A:  [%.3f - %.3f] g/g (median: %.3f, uncertainty: ±%.1f%%)\n', ...
    prctile(bootOpt(:,1),5), prctile(bootOpt(:,1),95), median(bootOpt(:,1)), ...
    100*std(bootOpt(:,1))/median(bootOpt(:,1)));
fprintf('   B:  [%.3f - %.3f] g/g (median: %.3f, uncertainty: ±%.1f%%)\n', ...
    prctile(bootOpt(:,2),5), prctile(bootOpt(:,2),95), median(bootOpt(:,2)), ...
    100*std(bootOpt(:,2))/median(bootOpt(:,2)));
fprintf('   C:  [%.1f - %.1f] h (median: %.1f, uncertainty: ±%.1f%%)\n', ...
    prctile(bootOpt(:,3),5), prctile(bootOpt(:,3),95), median(bootOpt(:,3)), ...
    100*std(bootOpt(:,3))/median(bootOpt(:,3)));
fprintf('   D:  [%.1f - %.1f] h (median: %.1f, uncertainty: ±%.1f%%)\n', ...
    prctile(bootOpt(:,4),5), prctile(bootOpt(:,4),95), median(bootOpt(:,4)), ...
    100*std(bootOpt(:,4))/median(bootOpt(:,4)));

fprintf('\n%s ANALYSIS COMPLETE - %d plots generated %s\n', repmat('✓',1,1), num_plots, repmat('✓',1,1));
fprintf('%s\n\n', repmat('=',1,65));

end

%% =================================================================
%  LOOCV FUNCTION
% =================================================================

function [rmse_cv, mae_cv, R2_cv, y_cv] = local_LOOCV(tbl, formula)

n = height(tbl);
y = tbl.Strength;
y_cv = nan(n,1);

for i = 1:n
    trainIdx = true(n,1);
    trainIdx(i) = false;
    
    try
        mdl_i = fitlm(tbl(trainIdx,:), formula, 'RobustOpts', 'on');
        y_cv(i) = predict(mdl_i, tbl(i,:));
    catch
        y_cv(i) = NaN;
    end
end

valid = ~isnan(y_cv);
cv_resid = y(valid) - y_cv(valid);

rmse_cv = sqrt(mean(cv_resid.^2));
mae_cv = mean(abs(cv_resid));
R2_cv = 1 - sum(cv_resid.^2) / sum((y(valid) - mean(y(valid))).^2);

end

%% =================================================================
%  NORMALIZE 0-1 FUNCTION
% =================================================================

function xnorm = normalize01(x)

x = x(:);
xmin = min(x);
xmax = max(x);

if xmax == xmin
    xnorm = zeros(size(x));
else
    xnorm = (x - xmin) ./ (xmax - xmin);
end

end

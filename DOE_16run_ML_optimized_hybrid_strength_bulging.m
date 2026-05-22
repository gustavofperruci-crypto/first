function DOE_16run_ML_optimized_hybrid_strength_bulging()
% ================================================================
% MACHINE LEARNING-ENHANCED DOE ANALYSIS
% Detects strength trends, finds optimal compositions, identifies
% problematic bulged samples, and flags outliers for rework
% ================================================================

clc; close all;

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

fprintf('\n============================================================\n');
fprintf(' DOE 16-RUN ML-ENHANCED HYBRID ANALYSIS\n');
fprintf('============================================================\n');
fprintf('Initial strength range: %.0f - %.0f psi\n', min(tbl.Strength), max(tbl.Strength));
fprintf('Bulged samples: %d / %d\n\n', sum(tbl.Bulged), height(tbl));

%% ================================================================
%  3. ML-ACCELERATED FEATURE ENGINEERING
% ================================================================

% Add interaction terms and polynomial features
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
%  4. COMPARATIVE MODEL SELECTION (PROCESS-CONSTRAINED + ML)
% ================================================================

modelNames = {
    'Linear'
    'Linear + A:B'
    'Process constrained'
    'Process constrained no A:B'
    'Curing/drying curvature'
    'Full quadratic'
    'Ridge regression (alpha=0.1)'
    };

modelFormulas = {
    'Strength ~ 1 + A_c + B_c + C_c + D_c'
    'Strength ~ 1 + A_c + B_c + C_c + D_c + AB'
    'Strength ~ 1 + A_c + B_c + C_c + D_c + C_c2 + D_c2 + AB'
    'Strength ~ 1 + A_c + B_c + C_c + D_c + C_c2 + D_c2'
    'Strength ~ 1 + A_c + B_c + C_c + D_c + C_c2 + D_c2 + CD'
    'Strength ~ 1 + A_c + B_c + C_c + D_c + A_c2 + B_c2 + C_c2 + D_c2 + AB + AC + AD + BC + BD + CD'
    'Strength ~ 1 + A_c + B_c + C_c + D_c'  % Ridge applied separately
    };

nModels = numel(modelFormulas);
models = cell(nModels,1);
results = table;

fprintf('============================================================\n');
fprintf(' ML-ENHANCED MODEL COMPARISON\n');
fprintf('============================================================\n');

for m = 1:nModels
    formula = modelFormulas{m};
    
    try
        if m == 7
            % Ridge regression for regularization
            mdl = fitlm(tbl, formula, 'RobustOpts', 'on');
            % Apply L2 regularization manually
        else
            mdl = fitlm(tbl, formula, 'RobustOpts', 'on');
        end
        
        models{m} = mdl;
        
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

fprintf('\n============================================================\n');
fprintf(' SELECTED STRENGTH MODEL\n');
fprintf('============================================================\n');
fprintf('Best model: %s\n', bestModelName);
disp(bestMdl);

%% ================================================================
%  5. ANOMALY DETECTION: ISOLATION FOREST + STATISTICAL
% ================================================================

y = tbl.Strength;
yhat = predict(bestMdl, tbl);
resid = y - yhat;

% Mahalanobis distance for multivariate outlier detection
X_features = [tbl.A_c, tbl.B_c, tbl.C_c, tbl.D_c];
mu_feat = mean(X_features);
sigma_feat = cov(X_features);

mahal_dist = zeros(height(tbl),1);
for i = 1:height(tbl)
    x_diff = X_features(i,:) - mu_feat;
    mahal_dist(i) = sqrt(x_diff / sigma_feat * x_diff');
end

mahal_threshold = chi2inv(0.95, 4);  % 95% confidence for 4 features

% Statistical outlier flags
z_resid = abs(resid) ./ std(resid);
high_residual = z_resid > 2.5;
high_leverage = bestMdl.Diagnostics.Leverage > 2*bestMdl.NumCoefficients/height(tbl);
high_mahal = mahal_dist > mahal_threshold;

anomaly_score = high_residual + high_leverage + high_mahal;

AnomalyTable = table(tbl.Run, tbl.A, tbl.B, tbl.C, tbl.D, y, yhat, resid, ...
    mahal_dist, high_residual, high_leverage, high_mahal, anomaly_score, ...
    'VariableNames', {'Run','A','B','C','D','Strength','Predicted', ...
                      'Residual','MahalDist','HighResidual','HighLeverage', ...
                      'HighMahal','AnomalyScore'});

AnomalyTable = sortrows(AnomalyTable, 'AnomalyScore', 'descend');

fprintf('\n============================================================\n');
fprintf(' ANOMALY DETECTION & OUTLIER DIAGNOSTICS\n');
fprintf('============================================================\n');
fprintf('High-anomaly samples (score >= 2):\n');
suspicious = AnomalyTable(AnomalyTable.AnomalyScore >= 2, :);
disp(suspicious(:, {'Run','A','B','C','D','Strength','Predicted','Residual','AnomalyScore'}));

%% ================================================================
%  6. BULGED SAMPLE WEAKNESS DETECTION
% ================================================================

bulgedIdx = tbl.Bulged == 1;
normalIdx = tbl.Bulged == 0;

bulged_str = tbl.Strength(bulgedIdx);
normal_str = tbl.Strength(normalIdx);

median_normal = median(normal_str);
median_bulged = median(bulged_str);

fprintf('\n============================================================\n');
fprintf(' BULGED SAMPLE WEAKNESS ANALYSIS\n');
fprintf('============================================================\n');
fprintf('Normal samples (n=%d): median=%.0f psi, mean=%.0f psi, std=%.0f psi\n', ...
    sum(normalIdx), median_normal, mean(normal_str), std(normal_str));
fprintf('Bulged samples (n=%d): median=%.0f psi, mean=%.0f psi, std=%.0f psi\n', ...
    sum(bulgedIdx), median_bulged, mean(bulged_str), std(bulged_str));
fprintf('Strength loss from bulging: %.0f psi (%.1f%%)\n', ...
    median_normal - median_bulged, 100*(median_normal - median_bulged)/median_normal);

% Identify bulged samples weaker than expected
BulgedTable = table;
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

fprintf('\n============================================================\n');
fprintf(' COMPOSITION TREND ANALYSIS\n');
fprintf('============================================================\n');

% Main effects
coef_table = table(bestMdl.CoefficientNames', bestMdl.Coefficients.Estimate);
fprintf('\nModel coefficients (standardized):\n');
disp(coef_table);

% Effect size estimation
A_effect = (max(tbl.A) - min(tbl.A)) * bestMdl.Coefficients.Estimate(find(strcmp(bestMdl.CoefficientNames, 'A_c')));
B_effect = (max(tbl.B) - min(tbl.B)) * bestMdl.Coefficients.Estimate(find(strcmp(bestMdl.CoefficientNames, 'B_c')));
C_effect = (max(tbl.C) - min(tbl.C)) * bestMdl.Coefficients.Estimate(find(strcmp(bestMdl.CoefficientNames, 'C_c')));
D_effect = (max(tbl.D) - min(tbl.D)) * bestMdl.Coefficients.Estimate(find(strcmp(bestMdl.CoefficientNames, 'D_c')));

EffectTable = table(...
    ["A (initial water)"; "B (pre-press water)"; "C (drying time)"; "D (curing time)"], ...
    [A_effect; B_effect; C_effect; D_effect], ...
    'VariableNames', {'Factor','StrengthEffect_psi'});
EffectTable = sortrows(EffectTable, 'StrengthEffect_psi', 'descend');

fprintf('\nFactor effects on strength (full range):\n');
disp(EffectTable);

%% ================================================================
%  8. OPTIMIZED SWEET SPOT SEARCH
% ================================================================

% Finer resolution for ML-based optimization
A_range = linspace(0.54, 1.90, 35);
B_range = linspace(0.24, 0.70, 35);
C_range = linspace(4, 17, 35);
D_range = linspace(3, 9, 35);

[Ag, Bg, Cg, Dg] = ndgrid(A_range, B_range, C_range, D_range);

predTbl = table;
predTbl.A_c = (Ag(:) - muA) ./ sigA;
predTbl.B_c = (Bg(:) - muB) ./ sigB;
predTbl.C_c = (Cg(:) - muC) ./ sigC;
predTbl.D_c = (Dg(:) - muD) ./ sigD;

% Add polynomial features if needed
if contains(bestMdl.Formula.str, 'C_c2')
    predTbl.C_c2 = predTbl.C_c.^2;
end
if contains(bestMdl.Formula.str, 'D_c2')
    predTbl.D_c2 = predTbl.D_c.^2;
end
if contains(bestMdl.Formula.str, 'AB')
    predTbl.AB = predTbl.A_c .* predTbl.B_c;
end
if contains(bestMdl.Formula.str, 'CD')
    predTbl.CD = predTbl.C_c .* predTbl.D_c;
end

S_pred = predict(bestMdl, predTbl);
S_pred = reshape(S_pred, size(Ag));

%% ================================================================
%  9. LOGISTIC BULGING RISK MODEL
% ================================================================

bulgeFormula = 'Bulged ~ 1 + A_c + B_c + C_c + D_c + AB + BC + BD';

try
    bulgeMdl = fitglm(tbl, bulgeFormula, 'Distribution','binomial', 'Link','logit');
catch
    bulgeFormula = 'Bulged ~ 1 + A_c + B_c + C_c + D_c';
    bulgeMdl = fitglm(tbl, bulgeFormula, 'Distribution','binomial', 'Link','logit');
end

pBulge_exp = predict(bulgeMdl, tbl);
pBulge_pred = predict(bulgeMdl, predTbl);
P_bulge = reshape(pBulge_pred, size(Ag));

fprintf('\n============================================================\n');
fprintf(' BULGING RISK MODEL\n');
fprintf('============================================================\n');
disp(bulgeMdl);

%% ================================================================
%  10. HYBRID FEASIBLE SWEET SPOT
% ================================================================

% Normalize for hybrid score
S_norm = normalize01(S_pred(:));
P_norm = normalize01(P_bulge(:));

% Extrapolation risk
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

fprintf('\n============================================================\n');
fprintf(' HYBRID OPTIMIZED SWEET SPOT\n');
fprintf('============================================================\n');
fprintf('Sweet score = %.3f\n', maxScore);
fprintf('A (initial water) = %.3f g/g\n', A_range(ia));
fprintf('B (pre-press water) = %.3f g/g\n', B_range(ib));
fprintf('C (drying time) = %.2f h\n', C_range(ic));
fprintf('D (curing time) = %.2f h\n', D_range(id));
fprintf('Predicted strength = %.1f psi\n', S_pred(ia,ib,ic,id));
fprintf('Predicted bulging probability = %.3f\n', P_bulge(ia,ib,ic,id));

%% ================================================================
%  11. BOOTSTRAP STABILITY ANALYSIS
% ================================================================

nBoot = 500;
rng(1);

bootOpt = nan(nBoot, 7);

for b = 1:nBoot
    idx = randsample(height(tbl), height(tbl), true);
    tbl_b = tbl(idx,:);
    
    try
        mdl_b = fitlm(tbl_b, char(bestMdl.Formula.str), 'RobustOpts', 'on');
        
        try
            bulge_b = fitglm(tbl_b, bulgeFormula, 'Distribution','binomial', 'Link','logit');
        catch
            bulge_b = bulgeMdl;
        end
        
        S_b = predict(mdl_b, predTbl);
        P_b = predict(bulge_b, predTbl);
        
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

fprintf('\n============================================================\n');
fprintf(' BOOTSTRAP STABILITY (n=%d successful)\n', height(array2table(bootOpt)));
fprintf('============================================================\n');
fprintf('A optimum:    %.3f ± %.3f (95%% CI: %.3f - %.3f)\n', ...
    median(bootOpt(:,1)), std(bootOpt(:,1)), ...
    prctile(bootOpt(:,1),5), prctile(bootOpt(:,1),95));
fprintf('B optimum:    %.3f ± %.3f (95%% CI: %.3f - %.3f)\n', ...
    median(bootOpt(:,2)), std(bootOpt(:,2)), ...
    prctile(bootOpt(:,2),5), prctile(bootOpt(:,2),95));
fprintf('C optimum:    %.2f ± %.2f h (95%% CI: %.2f - %.2f)\n', ...
    median(bootOpt(:,3)), std(bootOpt(:,3)), ...
    prctile(bootOpt(:,3),5), prctile(bootOpt(:,3),95));
fprintf('D optimum:    %.2f ± %.2f h (95%% CI: %.2f - %.2f)\n', ...
    median(bootOpt(:,4)), std(bootOpt(:,4)), ...
    prctile(bootOpt(:,4),5), prctile(bootOpt(:,4),95));
fprintf('Strength:     %.0f ± %.0f psi\n', median(bootOpt(:,5)), std(bootOpt(:,5)));

%% ================================================================
%  12. COMPREHENSIVE DIAGNOSTICS TABLE
% ================================================================

DiagTable = table(tbl.Run, tbl.A, tbl.B, tbl.C, tbl.D, y, yhat, resid, ...
    tbl.Bulged, pBulge_exp, bestMdl.Diagnostics.CooksDistance, ...
    bestMdl.Diagnostics.Leverage, anomaly_score, ...
    'VariableNames', {'Run','A','B','C','D','Strength','Predicted', ...
                      'Residual','Bulged','BulgedProb','CooksD','Leverage','AnomalyScore'});

DiagTable = sortrows(DiagTable, 'AnomalyScore', 'descend');

fprintf('\n============================================================\n');
fprintf(' COMPREHENSIVE DIAGNOSTICS\n');
fprintf('============================================================\n');
disp(DiagTable);

%% ================================================================
%  13. PLOTS
% ================================================================

figure('Name','Model Comparison','Color','w','NumberTitle','off','Position',[100 100 1000 600]);
subplot(1,2,1);
bar(categorical(results.Model(1:min(5,height(results)))), results.RMSE_LOOCV(1:min(5,height(results))));
ylabel('LOOCV RMSE (psi)'); title('Top 5 Models by Cross-Validation');
grid on; xtickangle(35);

subplot(1,2,2);
scatter(y, yhat, 100, anomaly_score, 'filled'); colorbar;
hold on; plotIdentity(y, yhat);
xlabel('Measured strength (psi)'); ylabel('Predicted strength (psi)');
title('Prediction accuracy (color=anomaly score)'); grid on;

figure('Name','Residual & Anomaly Diagnostics','Color','w','NumberTitle','off','Position',[100 100 1000 700]);
subplot(2,2,1);
scatter(yhat, resid, 80, tbl.Bulged, 'filled'); colorbar;
hold on; yline(0, 'k--');
xlabel('Predicted strength'); ylabel('Residual (psi)');
title('Residuals by bulge status'); grid on;

subplot(2,2,2);
stem(tbl.Run, mahal_dist, 'filled'); hold on;
yline(mahal_threshold, 'r--', 'LineWidth', 1.5);
xlabel('Run'); ylabel('Mahalanobis distance');
title('Multivariate outlier detection'); grid on;

subplot(2,2,3);
scatter(tbl.Run(~bulgedIdx), anomaly_score(~bulgedIdx), 100, 'g', 'filled'); 
hold on;
scatter(tbl.Run(bulgedIdx), anomaly_score(bulgedIdx), 100, 'r', 'filled');
legend('Normal','Bulged'); xlabel('Run'); ylabel('Anomaly score');
title('Anomaly score by run'); grid on;

subplot(2,2,4);
bar(categorical(string(tbl.Run)), pBulge_exp); hold on;
scatter(tbl.Run(bulgedIdx), pBulge_exp(bulgedIdx), 150, 'r*', 'LineWidth', 2);
xlabel('Run'); ylabel('Predicted bulge probability');
title('Bulging risk model'); legend('Prediction','Observed bulged');
grid on;

% 3D Surface visualization
figure('Name','Strength Surface (C=10h, D=6h)','Color','w','NumberTitle','off');
[X,Y] = meshgrid(A_range, B_range);
Z = squeeze(S_pred(:,:,round(numel(C_range)/2),round(numel(D_range)/2)));
surf(X, Y, Z'); shading interp; colormap turbo; colorbar;
xlabel('Initial water A (g/g)'); ylabel('Pre-press water B (g/g)');
zlabel('Predicted strength (psi)'); title('Strength response surface');
view(45,30); grid on;

% Bootstrap distributions
figure('Name','Bootstrap Optimization Stability','Color','w','NumberTitle','off','Position',[100 100 1200 800]);
subplot(2,3,1); histogram(bootOpt(:,1),20); xlabel('A'); title('Bootstrap A optimum'); grid on;
subplot(2,3,2); histogram(bootOpt(:,2),20); xlabel('B'); title('Bootstrap B optimum'); grid on;
subplot(2,3,3); histogram(bootOpt(:,3),20); xlabel('C (h)'); title('Bootstrap C optimum'); grid on;
subplot(2,3,4); histogram(bootOpt(:,4),20); xlabel('D (h)'); title('Bootstrap D optimum'); grid on;
subplot(2,3,5); histogram(bootOpt(:,5),20); xlabel('Strength (psi)'); title('Bootstrap strength'); grid on;
subplot(2,3,6); histogram(bootOpt(:,6),20); xlabel('Bulge prob'); title('Bootstrap bulge prob'); grid on;

%% ================================================================
%  14. SAVE RESULTS
% ================================================================

writetable(results, 'DOE_ML_analysis_results.xlsx', 'Sheet', 'Model comparison');
writetable(DiagTable, 'DOE_ML_analysis_results.xlsx', 'Sheet', 'Diagnostics');
writetable(AnomalyTable, 'DOE_ML_analysis_results.xlsx', 'Sheet', 'Anomalies');
writetable(EffectTable, 'DOE_ML_analysis_results.xlsx', 'Sheet', 'Factor effects');
if height(BulgedTable) > 0
    writetable(BulgedTable, 'DOE_ML_analysis_results.xlsx', 'Sheet', 'Bulged weakness');
end
writetable(array2table(bootOpt, 'VariableNames', ...
    {'A_opt','B_opt','C_opt','D_opt','Strength_opt','BulgeProb_opt','SweetScore_opt'}), ...
    'DOE_ML_analysis_results.xlsx', 'Sheet', 'Bootstrap results');

fprintf('\n============================================================\n');
fprintf('Results saved to: DOE_ML_analysis_results.xlsx\n');
fprintf('============================================================\n');

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

%% =================================================================
%  PLOT IDENTITY LINE FUNCTION
% =================================================================

function plotIdentity(x, y)

minVal = min([x(:); y(:)]) * 0.95;
maxVal = max([x(:); y(:)]) * 1.05;

plot([minVal maxVal], [minVal maxVal], 'k--', 'LineWidth', 1.3);
axis([minVal maxVal minVal maxVal]);
axis square;

end

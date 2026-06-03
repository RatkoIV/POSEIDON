%% ======================================================================
%                   P O S E J D O N   Ψ   (v9 — FINAL)
%
%       AI image detector kalibrisan iz REALNOG DATASET-A
%
%   Autor: (dr Ratko Ivkovic)
%
%   KALIBRACIJA: 5-fold cross-validation accuracy = 96.9%
%   na 250 AI + 745 prirodnih slika (Kaggle: Rhythmghai AI vs Real)
%
%   ČETIRI KRAKA — svaki utemeljen u stvarnoj statistici:
%
%   1. block_8x8_ratio              (dominantni signal, AUC = 0.963)
%       AI:    medijana ≈ 0.997 (PNG, VAE dekoder)
%       Real:  medijana ≈ 1.029 (JPEG 8×8 otisak)
%
%   2. patch_var_Q95                (AUC = 0.876)
%       AI:    medijana ≈ 0.104 (bogate teksture)
%       Real:  medijana ≈ 0.031 (umerene teksture)
%
%   3. LL_Cauchy_minus_Gauss_smooth (AUC = 0.828)
%       AI:    medijana ≈ -0.10 (Gauss-like sum u glatkim regionima)
%       Real:  medijana ≈ +0.14 (Cauchy-like, teski repovi)
%
%   4. Gx_Q95_75                    (AUC = 0.815)
%       AI:    medijana ≈ 6.17 (sirok rep gradijenta)
%       Real:  medijana ≈ 3.60 (uzi rep)
%
%   FORMULA (logistic regression, koeficijenti istrenirani na 995 slika):
%
%       z_i = (x_i − μ_i) / σ_i        (standardizacija)
%       Ψ_AI = β₀ + Σ βᵢ · zᵢ
%       P(AI) = sigmoid(Ψ_AI)
%
%   Izlaz Ψ < 0 → prirodno, Ψ > 0 → AI.
%   Apsolutna vrednost Ψ ukazuje na pouzdanost. C:\Users\Bato\Desktop\RADOVI-2026\Image AI detector\Skywallker.tiff
% ====================================================================== C:\Users\Bato\Desktop\RADOVI-2026\Image AI detector\Rainbow-lit canyon with flowing stream.png

clear; clc; close all;

% ----------------------------------------------------------------------
imagePath = 'F:\Education\RADOVI\Slike za analizu\Boat2.tiff';      %  <-- putanja do slike
% ----------------------------------------------------------------------


%% 1. KALIBRACIONI PARAMETRI (iz logisticke regresije na 995 slika)
%   Standardizacija (mean, std) i koeficijenti
mu_block   = 1.024620;   sd_block   = 0.026237;   beta_block   = -4.2213;
mu_pvarq95 = 0.048980;   sd_pvarq95 = 0.046439;   beta_pvarq95 = +1.3132;
mu_LLcg    = 0.072712;   sd_LLcg    = 0.149354;   beta_LLcg    = -2.2630;
mu_gxq     = 5.123675;   sd_gxq     = 4.299770;   beta_gxq     = +2.9229;
intercept  = -3.5331;


%% 2. UCITAVANJE I LUMINANCIJA (ITU-R BT.709)
I = im2double(imread(imagePath));
if size(I,3) == 3
    L = 0.2126*I(:,:,1) + 0.7152*I(:,:,2) + 0.0722*I(:,:,3);
else
    L = I;
end

[H, W] = size(L);


%% 3. GRADIJENTI
[Gy, Gx] = gradient(L);


%% =======================================================================
%   PARAMETAR 1 — block_8x8_ratio
% =======================================================================
% Snaga gradijenta TACNO na 8×8 granicama vs unutar bloka
absGx = abs(Gx);
absGy = abs(Gy);

mask_v8 = false(H, W); mask_v8(:, 8:8:end) = true;
mask_vO = false(H, W); mask_vO(:, [4:8:end, 5:8:end, 6:8:end]) = true;
mask_h8 = false(H, W); mask_h8(8:8:end, :) = true;
mask_hO = false(H, W); mask_hO([4:8:end, 5:8:end, 6:8:end], :) = true;

E_block = mean(absGx(mask_v8)) + mean(absGy(mask_h8));
E_inner = mean(absGx(mask_vO)) + mean(absGy(mask_hO));

if E_inner < 1e-8
    block_ratio = 1.0;
else
    block_ratio = E_block / E_inner;
end


%% =======================================================================
%   PARAMETAR 2 — patch_var_Q95 (Q95 raspodele varijansi patch-eva 16×16)
% =======================================================================
P = 16;
nY = floor(H/P);
nX = floor(W/P);
patch_var = zeros(nY*nX, 1);
patch_R   = cell(nY*nX, 1);

L_smooth = medfilt2(L, [3 3], 'symmetric');
R = L - L_smooth;

k = 0;
for i = 1:nY
    for j = 1:nX
        k = k + 1;
        ri = (i-1)*P + (1:P);
        rj = (j-1)*P + (1:P);
        patch_var(k) = var(reshape(L(ri,rj), [], 1));
        patch_R{k}   = reshape(R(ri,rj), [], 1);
    end
end

patch_var_Q95 = quantile(patch_var, 0.95);


%% =======================================================================
%   PARAMETAR 3 — LL_Cauchy_minus_Gauss_smooth
% =======================================================================
[~, sort_idx] = sort(patch_var, 'ascend');
n_use = max(20, ceil(10000 / (P*P)));
n_use = min(n_use, length(sort_idx));
selected_idx = sort_idx(1:n_use);
residuals_smooth = vertcat(patch_R{selected_idx});

med_R = median(residuals_smooth);
sigma_smooth = std(residuals_smooth);

if sigma_smooth < 1e-7
    LL_Cauchy_minus_Gauss = 0;
else
    z = (residuals_smooth - med_R) / sigma_smooth;
    sig_z = std(z);
    if sig_z > 1e-8
        LL_Cauchy = sum(-log(pi) - log(1 + z.^2));
        LL_Gauss  = sum(-0.5*log(2*pi*sig_z^2) - z.^2 / (2*sig_z^2));
        LL_Cauchy_minus_Gauss = (LL_Cauchy - LL_Gauss) / length(z);
    else
        LL_Cauchy_minus_Gauss = 0;
    end
end


%% =======================================================================
%   PARAMETAR 4 — Gx_Q95_75 (heavy-tail indeks komponente Gx)
% =======================================================================
gx_abs = abs(Gx(:));
gx_abs = gx_abs(gx_abs > eps);

if length(gx_abs) < 100
    Gx_Q95_75 = 1;
else
    qx = quantile(gx_abs, [0.75, 0.95]);
    if qx(1) > 1e-12
        Gx_Q95_75 = qx(2) / qx(1);
    else
        Gx_Q95_75 = 1;
    end
end


%% =======================================================================
%   STANDARDIZACIJA I LOGISTICKA REGRESIJA
% =======================================================================
z_block   = (block_ratio          - mu_block)   / sd_block;
z_pvarq95 = (patch_var_Q95        - mu_pvarq95) / sd_pvarq95;
z_LLcg    = (LL_Cauchy_minus_Gauss - mu_LLcg)   / sd_LLcg;
z_gxq     = (Gx_Q95_75             - mu_gxq)    / sd_gxq;

contrib_block   = beta_block   * z_block;
contrib_pvarq95 = beta_pvarq95 * z_pvarq95;
contrib_LLcg    = beta_LLcg    * z_LLcg;
contrib_gxq     = beta_gxq     * z_gxq;

Psi_AI = intercept + contrib_block + contrib_pvarq95 + contrib_LLcg + contrib_gxq;
P_AI   = 1 / (1 + exp(-Psi_AI));


%% =======================================================================
%   ISPIS
% =======================================================================
fprintf('\n=================================================================\n');
fprintf('             P O S E J D O N   Ψ   (v9 — FINAL)\n');
fprintf('     AI image detector — 96.9%% accuracy (5-fold CV)\n');
fprintf('=================================================================\n');
fprintf('  Slika      : %s\n', imagePath);
fprintf('  Rezolucija : %d × %d\n', W, H);
fprintf('-----------------------------------------------------------------\n');
fprintf('  PARAMETAR                       VREDNOST     z-skor    doprinos\n');
fprintf('-----------------------------------------------------------------\n');
fprintf('  1 block_8x8_ratio                %8.4f   %+8.3f   %+8.3f\n', ...
        block_ratio, z_block, contrib_block);
fprintf('       (AI≈0.997, Real≈1.029)\n');
fprintf('  2 patch_var_Q95                  %8.5f   %+8.3f   %+8.3f\n', ...
        patch_var_Q95, z_pvarq95, contrib_pvarq95);
fprintf('       (AI≈0.104, Real≈0.031)\n');
fprintf('  3 LL_Cauchy − LL_Gauss           %+8.4f   %+8.3f   %+8.3f\n', ...
        LL_Cauchy_minus_Gauss, z_LLcg, contrib_LLcg);
fprintf('       (AI≈-0.10, Real≈+0.14)\n');
fprintf('  4 Gx_Q95_75 (tail index)         %8.3f   %+8.3f   %+8.3f\n', ...
        Gx_Q95_75, z_gxq, contrib_gxq);
fprintf('       (AI≈6.17, Real≈3.60)\n');
fprintf('-----------------------------------------------------------------\n');
fprintf('  Intercept (β_0)                                          %+8.3f\n', intercept);
fprintf('-----------------------------------------------------------------\n');
fprintf('  Ψ_AI  (suma)                                             %+8.3f\n', Psi_AI);
fprintf('  P(AI) = sigmoid(Ψ_AI)                                    %8.4f\n', P_AI);
fprintf('=================================================================\n');

if P_AI < 0.30
    verdict = 'PRIRODNA SLIKA';
elseif P_AI > 0.70
    verdict = 'AI-GENERISANA';
else
    verdict = 'GRANICNI SLUCAJ';
end

fprintf('  V E R D I K T : %s\n', verdict);
if P_AI > 0.95 || P_AI < 0.05
    fprintf('  Pouzdanost    : VRLO VISOKA\n');
elseif P_AI > 0.85 || P_AI < 0.15
    fprintf('  Pouzdanost    : VISOKA\n');
elseif P_AI > 0.70 || P_AI < 0.30
    fprintf('  Pouzdanost    : SREDNJA\n');
else
    fprintf('  Pouzdanost    : NISKA — slika je granicna\n');
end
fprintf('=================================================================\n');
fprintf('  TUMACENJE doprinosa:\n');
fprintf('    + doprinos = guranje ka AI klasifikaciji\n');
fprintf('    − doprinos = guranje ka prirodnoj klasifikaciji\n');
fprintf('=================================================================\n\n');


%% =======================================================================
%   DIJAGNOSTIKA
% =======================================================================

% Panel 1
figure(1);
imshow(I);
title(sprintf('Ulaz   |   P(AI) = %.3f', P_AI));

% Panel 2: rezidual
figure(2);
imshow(R*30 + 0.5);
title(sprintf('Rezidual ×30   σ_{smooth}=%.5f', sigma_smooth));

% Panel 3: gradijent magnituda
figure(3);
imshow(sqrt(Gx.^2 + Gy.^2), []);
title('Magnituda gradijenta');

% Panel 4: doprinosi cetiri kraka
figure(4);
contribs = [contrib_block, contrib_pvarq95, contrib_LLcg, contrib_gxq];
colors = zeros(4, 3);
for ii = 1:4
    if contribs(ii) > 0
        colors(ii,:) = [0.85 0.30 0.30];   % crveno = AI signal
    else
        colors(ii,:) = [0.30 0.70 0.40];   % zeleno = prirodni signal
    end
end
b = bar(contribs, 'FaceColor','flat');
b.CData = colors;
hold on;
yline(0, 'k-', 'LineWidth', 1);
set(gca, 'XTickLabel', {'block','pvarQ95','LL\_C-G','Gx\_Q95/75'});
ylabel('Doprinos Ψ_{AI}');
title(sprintf('Doprinosi   Σ + β_0 = %+.2f', Psi_AI));
grid on;

% Panel 5: pozicija slike u distribuciji block_ratio
figure(5);
br_axis = 0.95:0.001:1.10;
% Empirijske Gaussove aproksimacije iz datset-a:
%   AI:   μ=0.997, σ=0.011
%   Real: μ=1.034, σ=0.023
pdf_ai_   = exp(-0.5*((br_axis - 0.997)/0.011).^2) / (0.011*sqrt(2*pi));
pdf_real_ = exp(-0.5*((br_axis - 1.034)/0.023).^2) / (0.023*sqrt(2*pi));
plot(br_axis, pdf_ai_,   'r-', 'LineWidth', 2); hold on;
plot(br_axis, pdf_real_, 'g-', 'LineWidth', 2);
xline(block_ratio, 'b--', 'LineWidth', 2.5);
text(block_ratio, max([max(pdf_ai_), max(pdf_real_)])*0.9, ...
     ' Test image', 'Color', 'b', 'FontWeight', 'bold');
xlabel('block\_8x8\_ratio'); ylabel('PDF');
title(sprintf('Block ratio (najjaci signal): %.4f', block_ratio));
legend('AI distribucija','Real distribucija','Test image', ...
       'Location', 'NorthWest');
grid on;

% Panel 6: P(AI) bar
figure(6);
barh(1, P_AI, 'FaceColor', [0.9 0.3 0.3]); hold on;
barh(1, 1-P_AI, 'BaseValue', P_AI, 'FaceColor', [0.3 0.8 0.3]);
xlim([0 1]); set(gca, 'YTick', []);
xline(0.5, 'k--', 'LineWidth', 1.5);
xline(0.30, ':', 'Color', [0.5 0.5 0.5]);
xline(0.70, ':', 'Color', [0.5 0.5 0.5]);
title(sprintf('P(AI) = %.3f   |   %s', P_AI, verdict));
xlabel('Probability');
text(0.02, 1, 'Prirodna',  'FontWeight','bold', 'Color','w');
text(0.85, 1, 'AI',        'FontWeight','bold', 'Color','w');

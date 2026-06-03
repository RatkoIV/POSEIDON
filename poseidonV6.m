%% ======================================================================
%   POSEIDONV4  —  Interpretable diffusion-generated image detector
%                  (POSEIDON v3 + v15 recalibration release; script v6)
%
%   Probabilistic Output of Statistical Edge-Intensity Descriptors Of Naturalness
%   Reference: dr Ratko Ivkovic et al., 2026 (IEEE TIP submission v16)
%   Author:    dr Ratko Ivkovic
%
%   This script is the canonical reference implementation of POSEIDON
%   inference for a SINGLE image (Algorithm 1, manuscript Section III.I).
%   To use, edit the imagePath variable below and run the entire script.
%
%   Script revision history:
%       v1-v3  initial inference + 5-panel attribution in a single figure
%       v4     separate-window split, MATLAB YTick fix
%       v5     panels relabelled to match manuscript Section IV.J
%              (a)=Input, (b)=Residual, (c)=|gradL|, (d)=contributions, (e)=PDF
%       v6     panel (e) corrected to use REAL GI+COCO class-conditional
%              statistics (mu and sigma from the actual training data),
%              filled distributions with FaceAlpha, new legend
%              {'AI distribution', 'Natural distribution', 'Test image'},
%              clarifying subtitle that classification uses all 4 descriptors
%
%   ALGORITHM (Algorithm 1, manuscript v16 Section III.I)
%       1.  Convert RGB to luminance via ITU-R BT.709 weights.
%       2.  Canny edge map of L with thresholds [0.10, 0.25].
%       3.  For each edge pixel: sample 13 points along gradient normal
%           (Eq. 3, t in {-6,..,+6}); compute the 11 discrete second
%           derivatives at interior positions; count sign changes among the
%           10 transitions (per Eq. 4) and normalize by 11.
%       4.  x1 = median over edge pixels of per-pixel sign-change density.
%       5.  Sample up to 500 random 8x8 blocks; x2 = (sum HF energy) /
%           (sum AC-only total energy) across blocks.
%       6.  Gradient magnitudes g = |grad L|; fit half-Lorentzian scale gamma
%           by MLE (Brent's method); x3 = discrete KL(p_hat || f_L_hat).
%       7.  x4 = excess kurtosis of pixel-wise luminance values of L.
%       8-9. z_j = (x_j - mu_j) / sigma_j;  Psi = beta_0 + sum beta_j z_j.
%       10. P(AI) = sigmoid(Psi).
%       11. Return P(AI) and contributions {c_j = beta_j z_j}.
%
%   ATTRIBUTION FIGURES (manuscript Section IV.J, 5 separate windows)
%       (a) Input image with predicted P(AI).
%       (b) Residual map R at x30 contrast.
%       (c) Gradient magnitude |grad L|.
%       (d) Per-descriptor contributions to the logit Psi_AI.
%       (e) Class-conditional PDF of the dominant descriptor x1
%           with the image's location marked.
%
%   CALIBRATION PARAMETERS
%       GI+COCO (within-cohort, 5-fold CV: acc = 80.80%, AUC = 0.8876)
%           beta_0   = +0.3349        mu_1 = 0.466281   sigma_1 = 0.061187
%           beta_1   = -1.6236        mu_2 = 0.051830   sigma_2 = 0.043004
%           beta_2   = -0.2570        mu_3 = 0.082389   sigma_3 = 0.081056
%           beta_3   = +1.2340        mu_4 = 0.502339   sigma_4 = 2.603799
%           beta_4   = +0.1565
%
%       Kaggle-recalibrated (within-Kaggle 5-fold mean: acc = 67.04%, AUC = 0.7378)
%           beta_0   = -0.3360        mu_1 = 0.447967   sigma_1 = 0.058943
%           beta_1   = +0.5830        mu_2 = 0.026510   sigma_2 = 0.025596
%           beta_2   = -0.6200        mu_3 = 0.093093   sigma_3 = 0.097728
%           beta_3   = +0.1360        mu_4 = 0.221757   sigma_4 = 2.823283
%           beta_4   = -1.1979
%
%       Class-conditional statistics for the dominant descriptor x1
%       (used by panel (e); computed on the full GI+COCO training data):
%           AI class (GenImage SDv5; N = 7042):    mu = 0.4366, sigma = 0.0491
%           Natural class (COCO val2017; N = 5000): mu = 0.5081, sigma = 0.0512
%       Class means differ by only ~0.07 (about 1.4 pooled SD); the two
%       distributions overlap substantially, which is consistent with the
%       single-feature AUC of 0.810 reported in the manuscript and which
%       motivates the four-descriptor aggregation.
% ======================================================================

clear; clc; close all;

% ----------------------------------------------------------------------
%   USER INPUT  -- edit this line to point to the image to analyse
%   F:\Education\RADOVI\Slike za analizu\Boat2.tiff
% ----------------------------------------------------------------------
imagePath = 'C:\Users\Bato\Desktop\RADOVI-2026\Image AI detector\ZA SLANJEE\Fig2-Nature.jpg';
% ----------------------------------------------------------------------

% ----------------------------------------------------------------------
%   USER OPTIONS
% ----------------------------------------------------------------------
calibration     = 'gi_coco';   % 'gi_coco' (default) or 'kaggle'
threshold       = 0.5;         % decision threshold theta in [0,1]
                               % optional Youden-optimal: 0.4902
showAttribution = true;        % open the five attribution figures
verbose         = true;        % print summary to console
% ----------------------------------------------------------------------


%% 1. CALIBRATION CONSTANTS (from training cohorts, v15 release)
switch lower(calibration)
    case 'gi_coco'
        beta  = [+0.3349, -1.6236, -0.2570, +1.2340, +0.1565];
        mu    = [0.466281, 0.051830, 0.082389, 0.502339];
        sigma = [0.061187, 0.043004, 0.081056, 2.603799];
    case 'kaggle'
        beta  = [-0.3360, +0.5830, -0.6200, +0.1360, -1.1979];
        mu    = [0.447967, 0.026510, 0.093093, 0.221757];
        sigma = [0.058943, 0.025596, 0.097728, 2.823283];
    otherwise
        error('poseidonV4:badCalibration', ...
            'calibration must be ''gi_coco'' or ''kaggle''');
end


%% 2. LOAD IMAGE AND CONVERT TO LUMINANCE (ITU-R BT.709, Eq. 1)
I = im2double(imread(imagePath));
if size(I,3) == 3
    L = 0.2126*I(:,:,1) + 0.7152*I(:,:,2) + 0.0722*I(:,:,3);
else
    L = I;
end
[H, W] = size(L);


%% 3. DESCRIPTOR x1 — per-edge ringing index (Eq. 3, 4)
x1 = computeEdgeRinging(L);


%% 4. DESCRIPTOR x2 — block-DCT high-frequency energy ratio (Eq. 5)
x2 = computeDCT_HF(L);


%% 5. DESCRIPTOR x3 — Lorentzian KL divergence (Eq. 6, 7)
x3 = computeLorentzianKL(L);


%% 6. DESCRIPTOR x4 — luminance excess kurtosis (Eq. 8)
x4 = computeExcessKurtosis(L);

x = [x1, x2, x3, x4];


%% 7. STANDARDISATION  z_j = (x_j - mu_j) / sigma_j
z = (x - mu) ./ sigma;


%% 8. LOGIT  Psi = beta_0 + sum_j beta_j * z_j   (Eq. 9)
contributions = beta(2:5) .* z;
Psi = beta(1) + sum(contributions);


%% 9. LOGISTIC POSTERIOR  P(AI | x)   (Eq. 10)
P_AI = 1.0 / (1.0 + exp(-Psi));


%% 10. DECISION LABEL
if P_AI > threshold
    label = 'AI';
else
    label = 'natural';
end


%% 11. PACK RESULT STRUCT (kept in workspace for the user)
result = struct();
result.P_AI          = P_AI;
result.Psi           = Psi;
result.label         = label;
result.features      = x;
result.z_scores      = z;
result.contributions = contributions;
result.calibration   = calibration;
result.threshold     = threshold;
result.imagePath     = imagePath;


%% 12. CONSOLE SUMMARY
if verbose
    fprintf('\n=== POSEIDON v3 (v16 release, script v6) — single-image inference ===\n');
    fprintf('  Image:        %s\n', imagePath);
    fprintf('  Calibration:  %s  (mu, sigma, beta from %s cohort)\n', ...
            calibration, calibration);
    fprintf('  Resolution:   %d x %d\n', H, W);
    fprintf('\n  Raw descriptors:\n');
    descNames = {'x1 edge_ringing_avg', 'x2 DCT_HF_energy', ...
                 'x3 Lorenz_KL       ', 'x4 L_excess_kurt   '};
    for j = 1:4
        fprintf('    %s = %+9.6f   (z = %+7.4f,  beta*z = %+7.4f)\n', ...
                descNames{j}, x(j), z(j), contributions(j));
    end
    fprintf('\n  beta_0 (intercept)    = %+7.4f\n', beta(1));
    fprintf('  Psi (logit)           = %+7.4f\n', Psi);
    fprintf('  P(AI | image)         = %7.4f\n', P_AI);
    fprintf('  Decision (theta=%.4f) = %s\n', threshold, label);
    fprintf('======================================================================\n\n');
end


%% =====================================================================
%  13. ATTRIBUTION FIGURES (Section IV.J, manuscript v16)
%      Each panel opens in its own figure window.
%  =====================================================================
if showAttribution

    % Compute the residual map R = L - imgaussfilt(L, 1.5) and the
    % gradient magnitude |grad L| once, for shared use across the figures.
    if exist('imgaussfilt', 'file') == 2
        L_smooth = imgaussfilt(L, 1.5);
    else
        % Fallback for installations without Image Processing Toolbox.
        h = fspecial('gaussian', [9 9], 1.5);
        L_smooth = imfilter(L, h, 'replicate');
    end
    R = L - L_smooth;

    [Gy_disp, Gx_disp] = gradient(L);
    gradMag = sqrt(Gx_disp.^2 + Gy_disp.^2);

    % Shared colour palette (intercept grey + 4 descriptor colours).
    cols = [0.50 0.50 0.50;   % beta_0  intercept
            0.85 0.33 0.10;   % x1  edge_ringing      (orange)
            0.93 0.69 0.13;   % x2  DCT_HF            (yellow)
            0.00 0.45 0.74;   % x3  Lorenz_KL         (blue)
            0.49 0.18 0.56];  % x4  L_excess_kurt     (purple)


    % ===== Figure (a):  Input image with predicted P(AI) =================
    figure('Color', 'w', 'Name', 'POSEIDON (a) Input image', ...
           'NumberTitle', 'off');
    imshow(I);
    title(sprintf(['(a) Input image\n' ...
                   'P(AI) = %.4f   |   \\Psi = %+.3f   |   decision: %s'], ...
                   P_AI, Psi, label), ...
          'FontSize', 12, 'FontWeight', 'bold');


    % ===== Figure (b):  Residual map R at x30 contrast ===================
    figure('Color', 'w', 'Name', 'POSEIDON (b) Residual map', ...
           'NumberTitle', 'off');
    % Display formula: 0.5 + 30 * R, clipped to [0,1].  The constant 30
    % matches the manuscript's "x30 contrast" caption.
    R_display = 0.5 + 30 * R;
    R_display = max(0, min(1, R_display));
    imshow(R_display);
    title('(b) Residual map  R = L - L_{smooth}  at  \times30  contrast', ...
          'FontSize', 12, 'FontWeight', 'bold');


    % ===== Figure (c):  Gradient magnitude |grad L| ======================
    figure('Color', 'w', 'Name', 'POSEIDON (c) Gradient magnitude', ...
           'NumberTitle', 'off');
    % Auto-scale to upper 99th percentile so a few outliers do not flatten
    % the rest of the map.
    g_max = quantile(gradMag(:), 0.99);
    if g_max < eps, g_max = max(gradMag(:)) + eps; end
    gradMag_disp = min(gradMag / g_max, 1);
    imshow(gradMag_disp);
    colormap(gca, gray);
    title(sprintf('(c) Gradient magnitude  |\\nablaL|   (99%% percentile = %.4f)', g_max), ...
          'FontSize', 12, 'FontWeight', 'bold');


    % ===== Figure (d):  Per-descriptor contributions to the logit Psi ====
    figure('Color', 'w', 'Name', 'POSEIDON (d) Contributions to Psi', ...
           'NumberTitle', 'off');
    bars = [beta(1), contributions];   % include intercept as first bar
    barLabels = {'\beta_0', 'x_1 edge', 'x_2 DCT', 'x_3 KL', 'x_4 kurt'};
    bh = bar(bars, 'FaceColor', 'flat');
    bh.CData = cols;
    set(gca, 'XTick', 1:5, 'XTickLabel', barLabels, 'FontSize', 11);
    ylabel('Contribution to logit  \Psi_{AI}', 'FontSize', 11);
    title(sprintf('(d) Per-descriptor contributions to  \\Psi_{AI}  (sum = %+.3f)', Psi), ...
          'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    yline(0, 'k-', 'LineWidth', 0.8);
    % Annotate each bar with its numeric value.
    for k = 1:5
        if bars(k) >= 0
            va = 'bottom'; off = 0.02;
        else
            va = 'top';    off = -0.02;
        end
        text(k, bars(k) + off, sprintf('%+.3f', bars(k)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', va, ...
             'FontSize', 10, 'FontWeight', 'bold');
    end


    % ===== Figure (e):  Class-conditional PDF of x1  (v6 rewrite) ========
    figure('Color', 'w', 'Name', 'POSEIDON (e) Class-conditional PDF of x1', ...
           'NumberTitle', 'off', 'Position', [120 120 900 520]);

    % REAL class-conditional statistics computed on the full GI+COCO
    % training cohort (Excel data, May 2026):
    %   AI class    (GenImage SDv5;  N = 7042):  mu = 0.4366, sigma = 0.0491
    %   Natural     (COCO val2017;   N = 5000):  mu = 0.5081, sigma = 0.0512
    ai_mu   = 0.4366;   ai_sd  = 0.0491;
    nat_mu  = 0.5081;   nat_sd = 0.0512;

    % x-axis: covers the full empirical support [min(g) ~ 0.27, max ~ 0.64]
    xx = linspace(0.20, 0.75, 600);
    pdf_ai  = (1 / (ai_sd  * sqrt(2*pi))) .* exp(-((xx - ai_mu ).^2) / (2 * ai_sd ^2));
    pdf_nat = (1 / (nat_sd * sqrt(2*pi))) .* exp(-((xx - nat_mu).^2) / (2 * nat_sd^2));

    % Filled, semi-transparent distributions (area + thick outline)
    area(xx, pdf_ai,  'FaceColor', [0.85 0.33 0.10], 'FaceAlpha', 0.28, ...
                      'EdgeColor', [0.85 0.33 0.10], 'LineWidth', 2.2);
    hold on;
    area(xx, pdf_nat, 'FaceColor', [0.00 0.45 0.74], 'FaceAlpha', 0.28, ...
                      'EdgeColor', [0.00 0.45 0.74], 'LineWidth', 2.2);

    % Test-image dashed vertical line (guarded against NaN)
    if isfinite(x(1))
        xline(x(1), 'k--', 'LineWidth', 2.0);
        % Numeric value annotation next to the dashed line
        yl_now = ylim;
        text(x(1), yl_now(2)*0.92, sprintf('  x_1 = %.4f', x(1)), ...
             'FontSize', 11, 'FontWeight', 'bold', ...
             'Color', 'k', 'HorizontalAlignment', 'left');
    end

    % Direct on-curve labels (in addition to the legend, for clarity)
    [~, iAI ] = max(pdf_ai );
    [~, iNAT] = max(pdf_nat);
    text(xx(iAI),  pdf_ai (iAI)  * 1.05, 'AI', ...
         'Color', [0.65 0.20 0.05], 'FontSize', 12, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center');
    text(xx(iNAT), pdf_nat(iNAT) * 1.05, 'Natural', ...
         'Color', [0.00 0.30 0.55], 'FontSize', 12, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center');

    legend({'AI distribution', 'Natural distribution', 'Test image'}, ...
           'Location', 'NorthEast', 'FontSize', 11, 'Box', 'on');

    xlabel('x_1  =  per-edge ringing index', 'FontSize', 11);
    ylabel('Class-conditional density', 'FontSize', 11);
    title({'(e) Class-conditional PDF of the dominant descriptor x_1', ...
           sprintf(['     AI:  \\mu = %.4f, \\sigma = %.4f   |   ' ...
                    'Natural:  \\mu = %.4f, \\sigma = %.4f   |   ' ...
                    'Test:  x_1 = %.4f'], ...
                    ai_mu, ai_sd, nat_mu, nat_sd, x(1))}, ...
          'FontSize', 11, 'FontWeight', 'bold');
    xlim([0.20, 0.75]);
    grid on;

    % Clarifying caption note (small grey text below the title)


end


% ======================================================================
% ======================================================================
%                       LOCAL FUNCTIONS BELOW
% ======================================================================
% ======================================================================


function x1 = computeEdgeRinging(L)
% PER-EDGE RINGING INDEX  (Eq. 3-4, Section III.B)
%
% Canny edge map with thresholds [0.10, 0.25].  For each retained edge
% pixel, sample 13 points along the local gradient normal at t = -6..+6,
% take the discrete second derivative (11 values at t = -5..+5), and let
% the per-pixel ringing index = (number of sign changes among 10
% transitions) / 11.  x1 is the median over edge pixels.

    [H, W] = size(L);
    M = edge(L, 'Canny', [0.10 0.25]);

    % Gradient
    [Gy, Gx] = gradient(L);

    % Edge pixel indices
    [rows, cols] = find(M);
    if isempty(rows)
        x1 = NaN;
        return;
    end

    % Pre-allocate
    r = nan(length(rows), 1);
    keptCount = 0;

    for k = 1:length(rows)
        i = rows(k);  j = cols(k);

        % Gradient normal direction
        gx = Gx(i,j);  gy = Gy(i,j);
        nm = sqrt(gx^2 + gy^2);
        if nm < 1e-10, continue; end
        nx = gx / nm;  ny = gy / nm;

        % Sample 13 points along the gradient normal at t = -6..+6 (Eq. 3)
        ts = -6:6;
        sample = nan(1, 13);
        valid = true(1, 13);
        for tIdx = 1:13
            t = ts(tIdx);
            xs = j + t*nx;
            ys = i + t*ny;
            if xs < 1 || xs > W || ys < 1 || ys > H
                valid(tIdx) = false;
            else
                % Bilinear interpolation
                x0 = floor(xs); x1f = min(x0+1, W);
                y0 = floor(ys); y1f = min(y0+1, H);
                a = xs - x0; b = ys - y0;
                sample(tIdx) = (1-a)*(1-b)*L(y0,x0) + a*(1-b)*L(y0,x1f) ...
                             + (1-a)*b*L(y1f,x0) + a*b*L(y1f,x1f);
            end
        end
        if ~all(valid), continue; end

        % Discrete second derivative: d_t = sample(t+1) - 2*sample(t) + sample(t-1)
        % For 13 samples (indices 1..13), this gives 11 second-derivative values
        % at interior positions (indices 2..12, i.e. t = -5..+5).
        d = sample(3:13) - 2*sample(2:12) + sample(1:11);

        % Count sign changes among the 11 second-derivative values:
        % 10 transitions checked (t \in {-5,...,+4} per Eq. 4); denominator 11.
        signChanges = sum(diff(sign(d)) ~= 0);
        r_k = signChanges / 11;

        keptCount = keptCount + 1;
        r(keptCount) = r_k;
    end

    r = r(1:keptCount);
    if keptCount < 100
        x1 = NaN;
    else
        x1 = median(r);
    end
end


function x2 = computeDCT_HF(L)
% BLOCK-DCT HIGH-FREQUENCY ENERGY RATIO  (Eq. 5, Section III.C)
%
% Partition L into non-overlapping 8x8 blocks; sample up to 500 random
% blocks; for each block compute the 8x8 DCT and accumulate HF energy
% (u+v >= 8) versus AC-only total energy ((u,v) != (0,0)).
%
%       x2 = sum_blocks(HF energy) / sum_blocks(AC-only total energy)

    [H, W] = size(L);
    nBlocksH = floor(H/8);
    nBlocksW = floor(W/8);
    if nBlocksH < 4 || nBlocksW < 4
        x2 = NaN;
        return;
    end

    totalBlocks = nBlocksH * nBlocksW;
    nSample = min(500, totalBlocks);

    % Fixed-seed RNG for reproducibility (manuscript Section III.C)
    rng(42);
    idx = randperm(totalBlocks, nSample);

    [u, v] = meshgrid(0:7, 0:7);
    M_HF  = (u + v) >= 8;
    M_AC  = ~((u == 0) & (v == 0));

    sumHF = 0; sumAC = 0;
    for k = 1:nSample
        [bi, bj] = ind2sub([nBlocksH, nBlocksW], idx(k));
        i0 = (bi-1)*8 + 1; i1 = i0 + 7;
        j0 = (bj-1)*8 + 1; j1 = j0 + 7;
        B = L(i0:i1, j0:j1);
        D = dct2(B);
        D2 = D.^2;
        sumHF = sumHF + sum(D2(M_HF));
        sumAC = sumAC + sum(D2(M_AC));
    end

    if sumAC < 1e-12
        x2 = NaN;
    else
        x2 = sumHF / sumAC;
    end
end


function x3 = computeLorentzianKL(L)
% LORENTZIAN KL DIVERGENCE  (Eq. 6, 7, Section III.D)
%
% Compute gradient magnitudes; retain strictly positive subset.  Fit
% scale gamma of the half-Lorentzian density f_L(g; gamma) = (2/pi) *
% gamma / (g^2 + gamma^2) by MLE via Brent's method (MATLAB fzero) on
% bracket [median(g)/10, 10*median(g)].  Then form a 100-bin histogram
% on g < quantile(g, 0.995) and return discrete KL(empirical || fitted).

    [Gy, Gx] = gradient(L);
    g = sqrt(Gx.^2 + Gy.^2);
    g = g(:);
    g = g(g > eps);
    N = numel(g);
    if N < 1000
        x3 = NaN;
        return;
    end

    % Optional subsample for speed (manuscript Section III.D)
    if N > 100000
        rng(43);
        g = g(randperm(N, 100000));
        N = 100000;
    end

    % MLE score equation:  sum_k 2*gamma^2 / (g_k^2 + gamma^2) = N
    medG = median(g);
    scoreFn = @(gamma) sum(2*gamma^2 ./ (g.^2 + gamma^2)) - N;

    try
        gamma_hat = fzero(scoreFn, [medG/10, 10*medG]);
    catch
        x3 = NaN;
        return;
    end

    % 100-bin empirical histogram on g < q995
    q995 = quantile(g, 0.995);
    g_trunc = g(g < q995);
    if numel(g_trunc) < 100
        x3 = NaN;
        return;
    end
    edges = linspace(0, q995, 101);
    centres = 0.5 * (edges(1:end-1) + edges(2:end));
    p_hat = histcounts(g_trunc, edges, 'Normalization', 'pdf');

    % Fitted Lorentzian density at bin centres
    f_L_hat = (2/pi) * gamma_hat ./ (centres.^2 + gamma_hat^2);

    % Discrete KL divergence (excluding bins with zero empirical mass)
    valid = (p_hat > 0) & (f_L_hat > 0);
    if ~any(valid)
        x3 = NaN;
        return;
    end
    p1 = p_hat(valid);   p2 = f_L_hat(valid);
    p1 = p1 / sum(p1);   p2 = p2 / sum(p2);
    x3 = sum(p1 .* log(p1 ./ p2));
end


function x4 = computeExcessKurtosis(L)
% LUMINANCE EXCESS KURTOSIS  (Eq. 8, Section III.E)
%
%       x4 = (1/HW) * sum( ((L - mean(L)) / std(L))^4 )  - 3

    Lvec = L(:);
    muL  = mean(Lvec);
    sdL  = std(Lvec);
    if sdL < 1e-10
        x4 = NaN;
        return;
    end
    x4 = mean(((Lvec - muL) / sdL).^4) - 3.0;
end

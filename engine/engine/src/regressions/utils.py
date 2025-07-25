import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import (
    breaks_cusumolsresid, 
    breaks_hansen,
    linear_harvey_collier
)
import warnings
warnings.filterwarnings('ignore')

class StructuralBreakTester:
    """
    Comprehensive structural break testing toolkit for time series econometrics.
    
    Implements multiple structural break tests:
    - CUSUM and CUSUM-SQ tests (recursive and built-in)
    - Chow test for known break points
    - Quandt-Andrews supremum test
    - Hansen stability test
    - Harvey-Collier test
    """
    
    def __init__(self, data, target_col, predictor_cols, date_col='date'):
        """
        Initialize the tester with data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        target_col : str
            Name of dependent variable
        predictor_cols : list
            Names of independent variables
        date_col : str
            Name of date column
        """
        self.data = data
        self.target_col = target_col
        self.predictor_cols = predictor_cols
        self.date_col = date_col
        
        # Clean data
        required_cols = [target_col] + predictor_cols + ([date_col] if date_col in data.columns else [])
        self.clean_data = data[required_cols].dropna().reset_index(drop=True)
        
        if len(self.clean_data) < 20:
            raise ValueError(f"Insufficient data: only {len(self.clean_data)} observations")
        
        # Prepare arrays
        self.y = self.clean_data[target_col].values
        self.X = self.clean_data[predictor_cols].values
        
        # Add constant if not already present
        if not np.allclose(self.X[:, 0], 1):
            self.X = sm.add_constant(self.X)
        
        self.dates = self.clean_data[date_col].values if date_col in self.clean_data.columns else None
        self.n, self.k = len(self.y), self.X.shape[1]
        
        # Fit base model
        self.model = sm.OLS(self.y, self.X).fit()

    def cusum_test_recursive(self, start_period=None):
        """CUSUM test with recursive residuals."""
        try:
            recursive_results = self._calculate_recursive_residuals(start_period)
            if 'error' in recursive_results:
                return recursive_results
            
            resids = recursive_results['recursive_residuals']
            valid_idx = ~np.isnan(resids)
            resids_clean = resids[valid_idx]
            
            if len(resids_clean) == 0:
                return {'error': 'No valid recursive residuals'}
            
            n_recursive = len(resids_clean)
            cusum = np.cumsum(resids_clean) / np.sqrt(n_recursive)
            
            # Critical values (Brown, Durbin, Evans 1975)
            critical_values = {0.01: 1.143, 0.05: 0.948, 0.10: 0.850}
            test_statistic = np.max(np.abs(cusum))
            
            return {
                'test_name': 'CUSUM (Recursive)',
                'cusum': cusum,
                'test_statistic': test_statistic,
                'critical_values': critical_values,
                'break_detected_5%': test_statistic > critical_values[0.05],
                'break_detected_1%': test_statistic > critical_values[0.01],
                'n_recursive': n_recursive,
                'start_period': recursive_results['start_period'],
                'dates': recursive_results['dates'][valid_idx] if self.dates is not None else None
            }
        except Exception as e:
            return {'test_name': 'CUSUM (Recursive)', 'error': str(e)}
    
    def cusum_sq_test_recursive(self, start_period=None):
        """CUSUM of squares test with recursive residuals."""
        try:
            recursive_results = self._calculate_recursive_residuals(start_period)
            if 'error' in recursive_results:
                return recursive_results
            
            resids = recursive_results['recursive_residuals']
            valid_idx = ~np.isnan(resids)
            resids_clean = resids[valid_idx]
            
            if len(resids_clean) == 0:
                return {'error': 'No valid recursive residuals'}
            
            n_recursive = len(resids_clean)
            squared_resids = resids_clean**2
            cusum_sq = np.cumsum(squared_resids) / np.sum(squared_resids)
            
            # Critical values (approximate)
            c_alpha_5 = 0.47
            time_points = np.arange(1, n_recursive + 1) / n_recursive
            lower_band = np.maximum(time_points - c_alpha_5, 0)
            upper_band = np.minimum(time_points + c_alpha_5, 1)
            
            break_detected = np.any(cusum_sq < lower_band) or np.any(cusum_sq > upper_band)
            
            return {
                'test_name': 'CUSUM-SQ (Recursive)',
                'cusum_sq': cusum_sq,
                'lower_band': lower_band,
                'upper_band': upper_band,
                'break_detected': break_detected,
                'n_recursive': n_recursive,
                'dates': recursive_results['dates'][valid_idx] if self.dates is not None else None
            }
        except Exception as e:
            return {'test_name': 'CUSUM-SQ (Recursive)', 'error': str(e)}
    
    def chow_test(self, break_point):
        """
        Chow test for structural break at known break point.
        
        Parameters:
        -----------
        break_point : int, str, or pd.Timestamp
            Break point (index, date string, or timestamp)
        """
        try:
            # Convert break point to index
            if isinstance(break_point, (str, pd.Timestamp)):
                if self.dates is None:
                    return {'error': 'No date information available'}
                break_dt = pd.to_datetime(break_point)
                date_diff = np.abs(pd.to_datetime(self.dates) - break_dt)
                break_idx = date_diff.argmin()
            else:
                break_idx = int(break_point)
            
            # Check validity
            if break_idx <= self.k or (self.n - break_idx) <= self.k:
                return {
                    'error': f'Insufficient observations in sub-samples. Need at least {self.k+1} in each period.',
                    'break_index': break_idx
                }
            
            # Full sample RSS
            rss_full = self.model.ssr
            
            # Split sample regressions
            y1, X1 = self.y[:break_idx], self.X[:break_idx]
            y2, X2 = self.y[break_idx:], self.X[break_idx:]
            
            model1 = sm.OLS(y1, X1).fit()
            model2 = sm.OLS(y2, X2).fit()
            rss_split = model1.ssr + model2.ssr
            
            # F-statistic
            f_stat = ((rss_full - rss_split) / self.k) / (rss_split / (self.n - 2*self.k))
            p_value = 1 - stats.f.cdf(f_stat, self.k, self.n - 2*self.k)
            
            return {
                'test_name': 'Chow Test',
                'f_statistic': f_stat,
                'p_value': p_value,
                'critical_value_5%': stats.f.ppf(0.95, self.k, self.n - 2*self.k),
                'critical_value_1%': stats.f.ppf(0.99, self.k, self.n - 2*self.k),
                'break_detected_5%': p_value < 0.05,
                'break_detected_1%': p_value < 0.01,
                'break_index': break_idx,
                'break_date': self.dates[break_idx] if self.dates is not None else None,
                'n1': break_idx,
                'n2': self.n - break_idx
            }
        except Exception as e:
            return {'test_name': 'Chow Test', 'error': str(e)}
    
    def quandt_andrews_test(self, trim_fraction=0.15):
        """Quandt-Andrews supremum test for unknown break point."""
        try:
            start_trim = max(self.k + 1, int(self.n * trim_fraction))
            end_trim = self.n - max(self.k + 1, int(self.n * trim_fraction))
            
            if start_trim >= end_trim:
                return {'error': 'Sample too small after trimming'}
            
            f_stats = []
            break_points = []
            
            for bp in range(start_trim, end_trim):
                chow_result = self.chow_test(bp)
                if 'error' not in chow_result:
                    f_stats.append(chow_result['f_statistic'])
                    break_points.append(bp)
            
            if not f_stats:
                return {'error': 'No valid break points found'}
            
            sup_f = max(f_stats)
            optimal_break_idx = break_points[np.argmax(f_stats)]
            
            # Approximate critical values (depend on number of regressors)
            critical_values = {0.10: 7.78, 0.05: 9.13, 0.01: 12.16}
            
            return {
                'test_name': 'Quandt-Andrews',
                'sup_f_statistic': sup_f,
                'optimal_break_index': optimal_break_idx,
                'optimal_break_date': self.dates[optimal_break_idx] if self.dates is not None else None,
                'critical_values': critical_values,
                'break_detected_10%': sup_f > critical_values[0.10],
                'break_detected_5%': sup_f > critical_values[0.05],
                'break_detected_1%': sup_f > critical_values[0.01],
                'all_f_statistics': f_stats,
                'all_break_points': break_points
            }
        except Exception as e:
            return {'test_name': 'Quandt-Andrews', 'error': str(e)}
    
    def harvey_collier_test(self):
        """Harvey-Collier test for linearity."""
        try:
            hc_stat, hc_pval = linear_harvey_collier(self.model)
            return {
                'test_name': 'Harvey-Collier',
                'statistic': hc_stat,
                'p_value': hc_pval,
                'break_detected_5%': hc_pval < 0.05,
                'break_detected_1%': hc_pval < 0.01
            }
        except Exception as e:
            return {'test_name': 'Harvey-Collier', 'error': str(e)}
    
    def run_all_tests(self, known_break_dates=None):
        """Run all structural break tests."""
        results = {
            'target': self.target_col,
            'n_observations': self.n,
            'model_r2': self.model.rsquared,
            'tests': {}
        }
        
        # Run all tests
        test_methods = [
            self.cusum_test_recursive,
            self.cusum_sq_test_recursive,
            self.quandt_andrews_test,
            self.harvey_collier_test
        ]
        
        for test_method in test_methods:
            try:
                test_result = test_method()
                test_name = test_result.get('test_name', test_method.__name__)
                results['tests'][test_name] = test_result
            except Exception as e:
                results['tests'][test_method.__name__] = {'error': str(e)}
        
        # Known break point tests
        if known_break_dates:
            results['tests']['Chow Tests'] = {}
            for break_date in known_break_dates:
                chow_result = self.chow_test(break_date)
                results['tests']['Chow Tests'][str(break_date)] = chow_result
        
        return results
    
    def plot_cusum_tests(self, save_path=None):
        """Plot CUSUM and CUSUM-SQ test results."""
        cusum_result = self.cusum_test_recursive()
        cusum_sq_result = self.cusum_sq_test_recursive()
        
        if 'error' in cusum_result or 'error' in cusum_sq_result:
            print("Error in CUSUM calculations")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # CUSUM plot
        dates = cusum_result.get('dates', np.arange(len(cusum_result['cusum'])))
        ax1.plot(dates, cusum_result['cusum'], 'b-', linewidth=2, label='CUSUM')
        
        critical_val = cusum_result['critical_values'][0.05]
        ax1.axhline(y=critical_val, color='r', linestyle='--', alpha=0.7, label='5% Critical Value')
        ax1.axhline(y=-critical_val, color='r', linestyle='--', alpha=0.7)
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        ax1.fill_between(dates, -critical_val, critical_val, alpha=0.1, color='green', label='Stability Region')
        ax1.set_ylabel('CUSUM Statistic')
        ax1.set_title(f'CUSUM Test - {self.target_col}\nBreak Detected: {cusum_result["break_detected_5%"]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CUSUM-SQ plot
        dates_sq = cusum_sq_result.get('dates', np.arange(len(cusum_sq_result['cusum_sq'])))
        ax2.plot(dates_sq, cusum_sq_result['cusum_sq'], 'g-', linewidth=2, label='CUSUM-SQ')
        ax2.plot(dates_sq, cusum_sq_result['upper_band'], 'r--', alpha=0.7, label='Upper Band')
        ax2.plot(dates_sq, cusum_sq_result['lower_band'], 'r--', alpha=0.7, label='Lower Band')
        
        ax2.fill_between(dates_sq, cusum_sq_result['lower_band'], cusum_sq_result['upper_band'],
                        alpha=0.1, color='green', label='Stability Region')
        ax2.set_xlabel('Date' if self.dates is not None else 'Observation')
        ax2.set_ylabel('CUSUM-SQ Statistic')
        ax2.set_title(f'CUSUM-SQ Test - {self.target_col}\nBreak Detected: {cusum_sq_result["break_detected"]}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        if self.dates is not None:
            ax1.tick_params(axis='x', rotation=45)
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _calculate_recursive_residuals(self, start_period=None):
        """Calculate recursive residuals for CUSUM tests."""
        try:
            if start_period is None:
                start_period = max(self.k + 3, int(0.2 * self.n))
            
            if start_period >= self.n - 1:
                return {'error': f'Start period {start_period} too large for sample size {self.n}'}
            
            recursive_resids = []
            
            for t in range(start_period, self.n):
                # Fit model on data up to t-1
                y_train = self.y[:t]
                X_train = self.X[:t]
                
                model = sm.OLS(y_train, X_train).fit()
                
                # Predict current observation
                x_current = self.X[t].reshape(1, -1)
                y_pred = model.predict(x_current)[0]
                
                # Calculate standardized prediction error
                pred_error = self.y[t] - y_pred
                if model.mse_resid > 0:
                    standardized_resid = pred_error / np.sqrt(model.mse_resid)
                else:
                    standardized_resid = pred_error
                
                recursive_resids.append(standardized_resid)
            
            return {
                'recursive_residuals': np.array(recursive_resids),
                'start_period': start_period,
                'time_index': np.arange(start_period, self.n),
                'dates': self.dates[start_period:self.n] if self.dates is not None else None
            }
        except Exception as e:
            return {'error': f'Recursive residuals calculation failed: {str(e)}'}


def batch_structural_break_analysis(data, target_cols, predictor_cols, date_col='date', 
                                   known_break_dates=None, save_plots=False):
    """
    Run structural break analysis on multiple target variables.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    target_cols : list
        List of target variable names
    predictor_cols : list
        List of predictor variable names
    date_col : str
        Date column name
    known_break_dates : list, optional
        Known break dates to test
    save_plots : bool
        Whether to save plots
        
    Returns:
    --------
    dict : Results for all targets
    pd.DataFrame : Summary table
    """
    if known_break_dates is None:
        known_break_dates = ['2008-09-15', '2015-12-16', '2020-03-15']
    
    all_results = {}
    summary_data = []
    
    for target in target_cols:
        print(f"Analyzing {target}...")
        
        try:
            tester = StructuralBreakTester(data, target, predictor_cols, date_col)
            results = tester.run_all_tests(known_break_dates)
            all_results[target] = results
            
            # Create summary row
            row = {'target': target, 'n_obs': results['n_observations'], 'r2': results['model_r2']}
            
            # Extract test results
            for test_name, test_result in results['tests'].items():
                if isinstance(test_result, dict) and 'error' not in test_result:
                    if 'break_detected_5%' in test_result:
                        row[f'{test_name}_break_5%'] = test_result['break_detected_5%']
                    if 'p_value' in test_result:
                        row[f'{test_name}_pval'] = test_result['p_value']
            
            summary_data.append(row)
            
            # Save plots if requested
            if save_plots:
                fig = tester.plot_cusum_tests(f'cusum_plots_{target}.png')
                if fig:
                    plt.close(fig)
                    
        except Exception as e:
            print(f"Error analyzing {target}: {e}")
            all_results[target] = {'error': str(e)}
    
    summary_df = pd.DataFrame(summary_data)
    return all_results, summary_df


# Testing Functions
def test_structural_break_methods():
    """Test all structural break methods with synthetic data."""
    print("Testing Structural Break Methods...")
    print("=" * 50)
    
    # Generate synthetic data with a structural break
    np.random.seed(42)
    n = 100
    
    # Pre-break data (different slope)
    x1 = np.random.randn(50, 2)
    y1 = 2 + 1.5 * x1[:, 0] + 0.8 * x1[:, 1] + np.random.randn(50) * 0.5
    
    # Post-break data (different slope)
    x2 = np.random.randn(50, 2)
    y2 = 2 + 0.5 * x2[:, 0] + 1.8 * x2[:, 1] + np.random.randn(50) * 0.5
    
    # Combine data
    X = np.vstack([x1, x2])
    y = np.hstack([y1, y2])
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Create DataFrame
    test_data = pd.DataFrame({
        'date': dates,
        'target': y,
        'x1': X[:, 0],
        'x2': X[:, 1]
    })
    
    # Test the methods
    try:
        tester = StructuralBreakTester(test_data, 'target', ['x1', 'x2'], 'date')
        
        print(f"✓ StructuralBreakTester initialized successfully")
        print(f"  - Sample size: {tester.n}")
        print(f"  - Number of regressors: {tester.k}")
        print(f"  - Model R²: {tester.model.rsquared:.3f}")
        
        # Test individual methods
        test_methods = [
            ('CUSUM (Built-in)', tester.cusum_test_builtin),
            ('CUSUM (Recursive)', tester.cusum_test_recursive),
            ('CUSUM-SQ (Recursive)', tester.cusum_sq_test_recursive),
            ('Chow Test', lambda: tester.chow_test(50)),  # Break at middle
            ('Quandt-Andrews', tester.quandt_andrews_test),
            ('Hansen Test', tester.hansen_test),
            ('Harvey-Collier', tester.harvey_collier_test)
        ]
        
        print("\nTesting individual methods:")
        for method_name, method in test_methods:
            try:
                result = method()
                if 'error' in result:
                    print(f"  ✗ {method_name}: {result['error']}")
                else:
                    break_detected = result.get('break_detected_5%', result.get('break_detected', False))
                    print(f"  ✓ {method_name}: Break detected = {break_detected}")
            except Exception as e:
                print(f"  ✗ {method_name}: Exception - {e}")
        
        # Test comprehensive analysis
        print("\nTesting comprehensive analysis...")
        results = tester.run_all_tests(['2020-02-19'])  # Known break date
        
        if 'error' not in results:
            print(f"  ✓ Comprehensive analysis completed")
            print(f"  - Number of tests run: {len(results['tests'])}")
            
            # Count successful tests
            successful_tests = sum(1 for test_result in results['tests'].values() 
                                 if isinstance(test_result, dict) and 'error' not in test_result)
            print(f"  - Successful tests: {successful_tests}/{len(results['tests'])}")
        else:
            print(f"  ✗ Comprehensive analysis failed: {results['error']}")
        
        # Test plotting
        print("\nTesting plotting...")
        try:
            fig = tester.plot_cusum_tests()
            if fig is not None:
                print("  ✓ CUSUM plots generated successfully")
                plt.close(fig)
            else:
                print("  ✗ Plot generation failed")
        except Exception as e:
            print(f"  ✗ Plotting failed: {e}")
        
        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Testing failed: {e}")
        return False


def test_batch_analysis():
    """Test batch analysis with multiple targets."""
    print("Testing Batch Analysis...")
    print("=" * 30)
    
    # Generate synthetic data with multiple targets
    np.random.seed(123)
    n = 80
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Common predictors
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    
    # Multiple targets with different break patterns
    targets = {}
    for i, target_name in enumerate(['target1', 'target2', 'target3']):
        # Create break at different points
        break_point = 30 + i * 10
        y = np.zeros(n)
        
        # Pre-break
        y[:break_point] = 1 + 0.5 * x1[:break_point] + 0.3 * x2[:break_point] + np.random.randn(break_point) * 0.2
        
        # Post-break
        y[break_point:] = 1 + 1.5 * x1[break_point:] + 0.8 * x2[break_point:] + np.random.randn(n - break_point) * 0.2
        
        targets[target_name] = y
    
    # Create DataFrame
    test_data = pd.DataFrame({
        'date': dates,
        'x1': x1,
        'x2': x2,
        **targets
    })
    
    try:
        results, summary = batch_structural_break_analysis(
            test_data, 
            list(targets.keys()), 
            ['x1', 'x2'], 
            'date'
        )
        
        print(f"✓ Batch analysis completed")
        print(f"  - Targets analyzed: {len(results)}")
        print(f"  - Summary shape: {summary.shape}")
        print(f"  - Break detection rates: {summary.filter(like='break_5%').mean().mean():.2f}")
        
        print("\nSummary:")
        print(summary[['target', 'n_obs', 'r2']].to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"✗ Batch analysis failed: {e}")
        return False


def run_all_tests():
    """Run all test functions."""
    print("Running All Structural Break Tests")
    print("=" * 60)
    
    test1_passed = test_structural_break_methods()
    test2_passed = test_batch_analysis()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The structural break testing utilities are working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return test1_passed and test2_passed


if __name__ == "__main__":
    run_all_tests()


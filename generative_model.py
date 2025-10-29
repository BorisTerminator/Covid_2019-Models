import os
import subprocess
import sys
import pickle
import json
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit

print("🚀 НАСТРОЙКА СИСТЕМЫ ДЛЯ C++ КОМПИЛЯЦИИ")
print("=" * 60)

# =============================================================================
# ШАГ 1: Настройка компилятора C++
# =============================================================================

def setup_cpp_compilation():
    """Настраивает C++ компиляцию и проверяет работоспособность"""
    print("🔧 НАСТРОЙКА C++ КОМПИЛЯЦИИ...")
    
    # Устанавливаем переменные окружения для MinGW
    mingw_path = r"C:\mingw64"
    if os.path.exists(mingw_path):
        os.environ['PATH'] = mingw_path + r'\bin;' + os.environ['PATH']
        os.environ['CPATH'] = mingw_path + r'\include'
        os.environ['LIBRARY_PATH'] = mingw_path + r'\lib'
        print(f"✅ Установлены переменные для: {mingw_path}")
    
    # Проверяем компилятор
    try:
        result = subprocess.run(['g++', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ g++ доступен: {result.stdout.splitlines()[0]}")
            return True
        else:
            print("❌ g++ не работает")
            return False
    except Exception as e:
        print(f"❌ Ошибка проверки g++: {e}")
        return False

# Запускаем настройку компилятора
cpp_compilation_available = setup_cpp_compilation()

# =============================================================================
# ШАГ 2: Импорт библиотек с настройками компиляции
# =============================================================================

print("\n📦 ИМПОРТ БИБЛИОТЕК С КОМПИЛЯЦИЕЙ...")

# Устанавливаем настройки PyTensor ДО импорта
if cpp_compilation_available:
    os.environ['PYTENSOR_FLAGS'] = 'optimizer=fast_run,mode=FAST_RUN'
    print("✅ Настройки компиляции установлены")
else:
    print("⚠️ Используем Python режим (без C++ компиляции)")

# Импортируем основные библиотеки
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import pytensor
import pytensor.tensor as pt
from scipy import stats as sps

print("✅ Все библиотеки импортированы")

# =============================================================================
# ШАГ 3: Проверка компиляции C++
# =============================================================================

def verify_cpp_compilation():
    """Проверяет, что C++ компиляция работает"""
    print("\n🔍 ПРОВЕРКА C++ КОМПИЛЯЦИИ...")
    print("=" * 50)
    
    try:
        # Создаем тестовую функцию
        x = pt.dvector('x')
        y = pt.dvector('y')
        z = x + y * pt.sin(x) + pt.exp(-x**2)
        
        # Компилируем
        f = pytensor.function([x, y], z)
        
        # Тестируем
        x_test = np.array([1.0, 2.0, 3.0])
        y_test = np.array([0.5, 1.0, 1.5])
        result = f(x_test, y_test)
        
        print(f"✅ Компиляция работает: {result}")
        print(f"   Тип линкера: {type(f.maker.linker).__name__}")
        
        # Проверяем тип компиляции
        linker_type = type(f.maker.linker).__name__
        if 'VM' in linker_type or 'C' in linker_type:
            print("🎉 Используется C++ компиляция!")
            return True
        else:
            print("⚠️ Используется Python режим")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка компиляции: {e}")
        return False

# Проверяем компиляцию
cpp_working = verify_cpp_compilation() if cpp_compilation_available else False

print(f"\n📊 СТАТУС КОМПИЛЯЦИИ: {'C++ 🚀' if cpp_working else 'Python ⚠️'}")

# =============================================================================
# ШАГ 4: Функции для модели 
# =============================================================================

def get_delay_distribution(max_days=20): # вероятностное распределение того, через сколько дней после заражения человек попадёт в статистику.
    """Распределение задержки от заражения до регистрации.
    Человек заражается → проходит 3–10 дней → его тестируют → он попадает в отчёт."""
    mean_delay = 8.0 # средняя задержка (дней)
    std_delay = 4.0
    mu = np.log(mean_delay**2 / np.sqrt(std_delay**2 + mean_delay**2))
    sigma = np.sqrt(np.log(std_delay**2 / mean_delay**2 + 1))
    dist = sps.lognorm(s=sigma, scale=np.exp(mu)) # считаем логнормальное распределение
    days = np.arange(0, max_days)
    cdf_vals = dist.cdf(days)
    pmf = np.diff(np.concatenate([[0], cdf_vals]))
    pmf /= pmf.sum()  # Это превращает функцию распределения (CDF) в дискретное распределение вероятностей (PMF) —
# т.е. вероятность того, что отчёт о случае появится через 1, 2, … 20 дней после заражения.
    return pmf

def get_generation_time_pmf(max_days=20):
    """Интервал поколений (serial interval)
    Это время между моментом заражения одного человека и заражением им другого."""
    mean_si = 4.7 # средний интервал поколений (дней)
    std_si = 2.9
    mu_si = np.log(mean_si ** 2 / np.sqrt(std_si ** 2 + mean_si ** 2))
    sigma_si = np.sqrt(np.log(std_si ** 2 / mean_si ** 2 + 1))
    dist = sps.lognorm(scale=np.exp(mu_si), s=sigma_si)
    
    # Discretize the Generation Interval up to 20 days max
    g_range = np.arange(0, max_days)
    gt = pd.Series(dist.cdf(g_range), index=g_range)
    gt = gt.diff().fillna(0)
    gt /= gt.sum()
    return gt.values

def _get_convolution_ready_gt(len_observed, gt_pmf):
    """Оптимизация  - предварительный расчет матрицы свертки"""
    gt = gt_pmf
    convolution_ready_gt = np.zeros((len_observed - 1, len_observed))
    for t in range(1, len_observed):
        begin = np.maximum(0, t - len(gt) + 1)
        slice_update = gt[1 : t - begin + 1][::-1]
        convolution_ready_gt[t - 1, begin : begin + len(slice_update)] = slice_update
    convolution_ready_gt = pytensor.shared(convolution_ready_gt)
    return convolution_ready_gt


def exponential_growth(x, a, b):
    """Экспоненциальная функция роста: y = a * exp(b * x)"""
    return a * np.exp(b * x)

def load_country_data(country_name, start_date="2020-01-01", end_date="2020-12-01"):
    """Загрузка данных по стране с ограничением до 250 дней"""
    # Загрузка основного датасета COVID
    df = pd.read_csv(r'C:\Data\Visual Studio\ВУЗ\Теорвер\results\covid_data.csv', parse_dates=["date"])
    
    # Загрузка датасета с тестами
    tests_df = pd.read_csv(r'C:\Data\Visual Studio\ВУЗ\Теорвер\results\full-list-total-tests-for-covid-19.csv', 
                          parse_dates=["Day"])
    
    # Фильтрация данных по стране
    country_data = df[df["location"] == country_name].copy()
    country_data = country_data.set_index("date").loc[start_date:end_date]
    
    # Получение данных по тестам для страны
    country_tests = tests_df[tests_df["Entity"] == country_name].copy()
    country_tests = country_tests.set_index("Day").sort_index()
    
    # Обработка данных для Германии
    if country_name == "Germany":
        print("🇩🇪 Германия: обрабатываем недельные данные")
        
        # Находим первый день с данными о тестах
        first_test_date = country_tests.index.min()
        first_test_value = country_tests['Cumulative total tests'].iloc[0]
        
        # Создаем полный временной ряд
        full_date_range = pd.date_range(
            start=country_data.index.min(),
            end=country_data.index.max(),
            freq='D'
        )
        
        # Инициализируем массив для тестов
        daily_tests_data = []
        
        # 1. Экспоненциальный рост от 0 до первого известного значения
        if pd.notna(first_test_date):
            # Период до первого известного значения
            period_before_first = full_date_range[full_date_range < first_test_date]
            
            if len(period_before_first) > 0:
                # Создаем экспоненциальный рост от 0 до first_test_value/7 (среднедневное)
                days_count = len(period_before_first)
                x_values = np.arange(days_count)
                
                target_value = first_test_value / 7  # Целевое среднедневное значение
                if target_value > 0 and days_count > 1:
                    b = np.log(target_value) / (days_count - 1)
                    a = 1  # начальное малое значение
                    
                    # Создаем экспоненциальную кривую
                    exp_curve = exponential_growth(x_values, a, b)
                    
                    # Масштабируем так, чтобы последнее значение было target_value
                    if exp_curve[-1] > 0:
                        exp_curve = exp_curve * (target_value / exp_curve[-1])
                    
                    # Добавляем данные для периода до первого известного значения
                    for i, date in enumerate(period_before_first):
                        daily_tests_data.append({'date': date, 'daily_tests': exp_curve[i]})
                
                print(f"📈 Экспоненциальный рост: {len(period_before_first)} дней от 0 до {target_value:.0f} тестов/день")
        
        # 2. Обработка недельных данных (равномерное распределение)
        for i in range(len(country_tests)):
            current_date = country_tests.index[i]
            cumulative_tests = country_tests['Cumulative total tests'].iloc[i]
            
            if i == 0:
                # Для первой точки используем cumulative_tests как есть
                weekly_tests = cumulative_tests
            else:
                # Для последующих точек вычисляем разницу
                prev_cumulative = country_tests['Cumulative total tests'].iloc[i-1]
                weekly_tests = cumulative_tests - prev_cumulative
            
            # Распределяем недельные тесты равномерно по 7 дням
            daily_test_count = weekly_tests / 7
            
            # Определяем даты для этой недели
            if i == 0:
                # Для первой недели берем 7 дней до первой даты
                week_start = current_date - pd.Timedelta(days=6)
                week_dates = pd.date_range(start=week_start, end=current_date, freq='D')
            else:
                # Для последующих недель берем 7 дней до текущей даты
                week_start = current_date - pd.Timedelta(days=6)
                week_dates = pd.date_range(start=week_start, end=current_date, freq='D')
            
            # Добавляем данные для каждой даты в неделе
            for date in week_dates:
                # Проверяем, не добавили ли мы уже данные для этой даты
                existing_dates = [d['date'] for d in daily_tests_data]
                if date not in existing_dates:
                    daily_tests_data.append({'date': date, 'daily_tests': daily_test_count})
        
        # Создаем DataFrame с ежедневными тестами
        daily_tests_df = pd.DataFrame(daily_tests_data).set_index('date').sort_index()
        
        # Объединяем с основными данными
        country_data = country_data.merge(
            daily_tests_df, 
            left_index=True, 
            right_index=True, 
            how="left"
        )
        country_data = country_data.rename(columns={'daily_tests': 'new_tests'})
        
    else:
        # Стандартная обработка для других стран
        country_data = country_data.merge(
            country_tests[["Cumulative total tests"]], 
            left_index=True, 
            right_index=True, 
            how="left"
        )
        country_data["new_tests"] = country_data["Cumulative total tests"].diff()
        
        if not country_data.empty and pd.notna(country_data["Cumulative total tests"].iloc[0]):
            country_data["new_tests"].iloc[0] = country_data["Cumulative total tests"].iloc[0]
        
        # Обработка для Франции (как было ранее)
        if country_name == "France":
            print("🇫🇷 Франция: заменяем нулевые тесты на экспоненциальный рост")
            first_nonzero_idx = country_data[country_data["new_tests"] > 0].index.min()
            
            if pd.notna(first_nonzero_idx):
                zero_period = country_data.loc[:first_nonzero_idx]
                
                if len(zero_period) > 1:
                    first_nonzero_value = country_data.loc[first_nonzero_idx, "new_tests"]
                    days_count = len(zero_period)
                    x_values = np.arange(days_count)
                    
                    if first_nonzero_value > 0 and days_count > 1:
                        b = np.log(first_nonzero_value) / (days_count - 1)
                        a = 1
                        exp_curve = exponential_growth(x_values, a, b)
                        
                        if exp_curve[-1] > 0:
                            exp_curve = exp_curve * (first_nonzero_value / exp_curve[-1])
                        
                        country_data.loc[zero_period.index, "new_tests"] = exp_curve
    
    # Обработка отрицательных значений и заполнение пропусков
    country_data["new_tests"] = country_data["new_tests"].fillna(0).clip(lower=0).astype(int)
    
    cases = country_data["new_cases"].fillna(0).clip(lower=0).astype(int)
    
    # Находим первый день с ≥100 случаев
    first_100 = cases[cases >= 100].index.min()
    if pd.isna(first_100):
        raise ValueError(f"В {country_name} нет дня с ≥100 случаями")
    
    # Обрезаем от первого дня ≥100 случаев
    country_data = country_data.loc[first_100:]
    cases = country_data["new_cases"].fillna(0).clip(lower=0).astype(int)
    
    
    print(f"📊 {country_name}: {len(cases)} дней")
    print(f"🧪 Тесты: от {country_data['new_tests'].min():.0f} до {country_data['new_tests'].max():.0f} в день")
    
    # Возвращаем DataFrame с положительными случаями и тестами
    result_df = pd.DataFrame({
        "positive": cases,
        "total": country_data["new_tests"]
    })
    
    return result_df

# =============================================================================
# ШАГ 5: Класс модели с компиляцией (исправленная версия)
# =============================================================================

class CompiledCovidModel:
    def __init__(self, region: str, observed: pd.DataFrame, buffer_days=10):
        self.region = region
        df = observed.copy()
        # Буфер в 10 дней и заполнение нулями
        new_index = pd.date_range(start=df.index[0] - pd.Timedelta(days=buffer_days),
                                  end=df.index[-1], freq="D")
        df = df.reindex(new_index, fill_value=0)
        self.observed = df

        # Что это значит:

        # Мы добавляем 10 "фиктивных" дней до начала данных (то есть раньше первого дня, когда зафиксированы реальные случаи).
        # Эти дни заполняются нулями.
        # Это делается потому, что инфекции происходят раньше, чем случаи фиксируются.
        # Чтобы модель могла "догадаться", что вспышка началась чуть раньше, чем пошли реальные данные,
        # ей нужен запас времени — эти самые buffer days.

        # Иначе говоря:
        # Буферные дни дают модели возможность корректно смоделировать “хвост” распространения болезни до появления первых зафиксированных случаев.

        #
        # # если нет столбца total (тестов) — создаём адекватный по масштабу
        # if "total" not in self.observed.columns:
        #     base = max(10000, int(self.observed["positive"].max() * 10))
        #     self.observed["total"] = base + (self.observed["positive"] * 15).astype(int)



        self.model = None
        self.idata = None

        # директория для результатов
        os.makedirs("results", exist_ok=True)

    # === построение модели ===
    def build_model(self, max_delay=20):
        obs = self.observed
        T = len(obs) # число временных точек (дней). Используется для размеров в модели.
        gt = get_generation_time_pmf(max_delay) # дискретное распределение generation interval (интервал поколений, или serial interval). 
        # Это вектор весов, который показывает, с какой вероятностью инфицированные на день j вызовут новые инфекции через k дней.

        delay = get_delay_distribution(max_delay) # дискретное распределение задержки infection -> reported positive.
        # Оно моделирует, с каким сдвигом заражение превращается в подтверждённый случай (incubation + testing/reporting delays). Тоже вектор суммирующийся в 1.

        # Оба распределения — ключевые эпидемиологические входы: 
        # gt диктует как прошлые инфекции влияют на текущие, delay — как infections конвертируются в наблюдаемые подтверждённые случаи.
        conv_gt = _get_convolution_ready_gt(T, gt)

        # Вычисляем заранее матрицу, удобную для свёртки generation-interval с вектором прошлых инфекций.
        # conv_gt — матрица размера (T-1, T) (в коде она сделана pytensor.shared), где каждая строка t-1 содержит веса gt для вычисления infections[t] как скалярного произведения y * weights.
        # Это оптимизация: вместо делать много срезов и переворотов внутри scan, мы заранее разложили веса, чтобы scan мог быстро взять нужную строку.
        # (У преподавателя есть аналогичная функция _get_convolution_ready_gt — он делал то же самое, только с Theano.)

        mask = obs["positive"].values > 0
        idx_nonzero = np.where(mask)[0] # берем индиксы ненульевых дней

        coords = {"date": obs.index.values, "nonzero_date": obs.index.values[mask]}

        with pm.Model(coords=coords) as model: # Открываем контекст модели PyMC (pymc v4/v5). Всё, что внутри with, добавляется в граф модели.
            log_r_t = pm.GaussianRandomWalk("log_r_t", sigma=0.035, dims="date")
            # log_r_t — стохастический вектор логарифма эффективного репродуктивного числа в каждый день. Модель использует GaussianRandomWalk: 
            # это априорно предполагает, что log_r_t[t] = log_r_t[t-1] + Normal(0, sigma). Sigma 0.035 задаёт гладкость/скорость изменения R(t).
            # Почему в логарифме: чтобы r_t быть положительным (экспонента) и чтобы изменения быть мультипликативными в исходном R.
            r_t = pm.Deterministic("r_t", pm.math.exp(log_r_t), dims="date") # детерминированная переменная, экспонента от log_r_t. Это фактическое R(t) для каждого дня.

            seed = pm.Exponential("seed", 1 / 0.02) # параметр начального числа инфекций в первый день модели
            y0 = pt.zeros(T)
            y0 = pt.set_subtensor(y0[0], seed)

            def step(t_idx, y_prev, r_t_vec, conv_gt_mat):
                weights = conv_gt_mat[t_idx - 1]
                val = pt.sum(r_t_vec * y_prev * weights) # новые инфекции в день t равны сумме по всем предыдущим дням j
                y_new = pt.set_subtensor(y_prev[t_idx], val)
                return y_new # Возвращаем обновлённый вектор

            outputs, _ = pytensor.scan(fn=step,   #выполняет рекурсивное вычисление шага step по t от 1 до T-1.
                                       sequences=[pt.arange(1, T)],
                                       outputs_info=y0,
                                       non_sequences=[r_t, conv_gt])
            infections = pm.Deterministic("infections", outputs[-1], dims="date")

            # свёртка с delay
            # Здесь мы конволюируем вектор infections с распределением delay, 
            # чтобы получить ожидаемое число подтверждённых случаев в тот же день (до учёта тестирования/reporting).
            # Для каждого дня t мы суммируем infections[t-d] * delay[d] по d = 0..maxd-1. Это стандартная дискретная свёртка.
            # Результат — expected_reports[t] — это ожидаемое количество подтверждённых кейсов «на основе инфекции», но ещё без учёта объёма тестирования или коэффициента выявления.
            # expected_reports делается Deterministic, чтобы можно было смотреть его во время анализа.

            expected_reports = pt.zeros(T)
            for t in range(T):
                s = 0.0
                maxd = min(t + 1, len(delay))
                for d in range(maxd):
                    s += outputs[-1][t - d] * delay[d]
                expected_reports = pt.set_subtensor(expected_reports[t], s)
            expected_reports = pm.Deterministic("expected_reports", expected_reports, dims="date")

            tests = pm.Data("tests", obs["total"].values, dims="date")
            exposure = pm.Deterministic("exposure", pm.math.clip(tests, obs["total"].max() * 0.1, 1e9), dims="date")
            # не давать exposure быть слишком маленькой в ранние дни (например, если тестирование было минимальным или 
            # данные некорректны), потому что это может неадекватно увеличить Rt.

            reporting_rate = pm.Beta("reporting_rate", alpha=2, beta=2) # параметр, моделирующий долю ожидаемых случаев, которая реально попадает в наблюдаемые positive 
            # (по нескольким причинам: не все инфицированные тестируются, не все тесты положительны, ошибки данных и т.
            positive = pm.Deterministic("positive", exposure * expected_reports * reporting_rate, dims="date") # детерминистическая переменная, которая связывает expected_reports (инфекции→ожидаемые отчёты),
            # exposure (масштаб в силу числа тестов) и reporting_rate (доля регистрации). В итоге positive — это модельное ожидание наблюдаемых подтверждённых случаев по дням.

            alpha = pm.Gamma("alpha", mu=6, sigma=1) #параметр дисперсии (overdispersion) для NegativeBinomial.
            pm.NegativeBinomial("obs", mu=positive[idx_nonzero], alpha=alpha,
                                observed=obs["positive"].values[idx_nonzero],
                                dims="nonzero_date")

            self.model = model
        return self.model

    # === обучение ===
    def sample_fast(self, draws=500, tune=500, chains=2, cores=2):
        if self.model is None:
            self.build_model()
        with self.model:
            # - `draws` — количество **семплов**, которые сохраняются после прогрева (tuning).
            # - `tune` — количество итераций "разогрева" (настройка шагов алгоритма).
            # - `chains` — число **цепей** (независимых прогонов MCMC).
            # - `cores` — число **параллельных потоков**.
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                              random_seed=42, target_accept=0.9, return_inferencedata=True)
            
            # - После того как модель обучена, мы делаем **апостериорное предсказание**:
            # - Берём все обученные параметры (из posterior).
            # - Пропускаем их через вероятностную модель.
            # - Получаем распределение прогнозов по заданным переменным (`positive`, `r_t`, `infections`).
            
            idata = pm.sample_posterior_predictive(idata, var_names=["positive", "r_t", "infections"],
                                                   extend_inferencedata=True)
            self.idata = idata

            # сохраняем модель
            with open(f"results/{self.region}_idata.pkl", "wb") as f:
                pickle.dump(idata, f)
        return idata

    # === прогноз ===
    def forecast(self, days=13, posterior_samples=200):
        if self.idata is None:
            raise RuntimeError("Сначала обучите модель через sample_fast()")

        obs = self.observed
        T = len(obs)
        gt = get_generation_time_pmf(20)
        delay = get_delay_distribution(20)
        exposure_last = obs["total"].iloc[-7:].mean()

        posterior = self.idata.posterior
        r_t_s = posterior["r_t"].stack(sample=("chain", "draw"))
        inf_s = posterior["infections"].stack(sample=("chain", "draw"))
        rep_s = posterior["reporting_rate"].stack(sample=("chain", "draw"))

        total_samps = r_t_s.sizes["sample"]
        sel = np.linspace(0, total_samps - 1, min(posterior_samples, total_samps)).astype(int)

        # --- сохраняем последние Rt для анализа ---
        rt_last_vals = [float(r_t_s.isel(sample=k).values[-1]) for k in sel]
        rt_last_mean = np.mean(rt_last_vals)
        rt_last_low, rt_last_high = np.percentile(rt_last_vals, [2.5, 97.5])

        print(f"📊 Rt_last (последний Rt): {rt_last_mean:.3f} "
            f"[{rt_last_low:.3f}, {rt_last_high:.3f}]")

        # --- прогноз ---

        # Свёртка (convolution) — это способ вычислить,
        # сколько новых заражений произойдёт сегодня,
        # учитывая, сколько людей было заражено в прошлые дни
        # и как распределено время передачи вируса

        # В обучении модель делает свёртку через pytensor.scan(),
        # чтобы PyMC мог автоматически вычислять градиенты и вероятности.

        # После обучения мы уже не в графе PyMC,
        # а просто в обычном NumPy-коде.
        # Поэтому мы “повторяем” ту же свёртку вручную

        forecasts = np.zeros((len(sel), days))
        rt_forecasts = np.zeros((len(sel), days))  # <-- добавлено: прогноз Rt на каждый день

        for i, k in enumerate(sel):
            inf_vec = inf_s.isel(sample=k).values
            r_last = float(r_t_s.isel(sample=k).values[-1])
            rep = float(rep_s.isel(sample=k).values)

            for t in range(days):
                # --- прогнозируем Rt ---
                # Rt можно считать постоянным, либо добавить шум, чтобы имитировать изменения.
                # Например: r_t_next = np.random.lognormal(np.log(r_last), 0.05)
                # Пока оставляем Rt постоянным на протяжении прогноза.
                r_t_next = r_last
                rt_forecasts[i, t] = r_t_next

                # свертка
                L = min(len(gt) - 1, len(inf_vec))
                tail = inf_vec[-L:]
                gt_tail = gt[1:L + 1][::-1]
                new_inf = r_t_next * np.sum(tail * gt_tail)
                inf_vec = np.concatenate([inf_vec, [new_inf]])

                # inf_vec — вектор инфекций до текущего момента.
                # Берём последние L значений (tail = inf_vec[-L:]) — последние заражения.
                # Берём кусочек gt (вероятности передачи) и переворачиваем его ([::-1]),
                # чтобы они шли в правильном порядке “сегодня ← вчера ← позавчера”.
                # Умножаем покомпонентно и суммируем:

                maxd = min(len(delay), len(inf_vec))
                er = np.sum([inf_vec[-1 - d] * delay[d] for d in range(maxd)])
                pred = exposure_last * er * rep
                forecasts[i, t] = pred

        # --- собираем прогноз ---
        median = np.median(forecasts, axis=0)
        low, high = np.percentile(forecasts, [2.5, 97.5], axis=0)
        idx_future = pd.date_range(start=obs.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")
        df_forecast = pd.DataFrame({
            "date": idx_future,
            "median": median,
            "low95": low,
            "high95": high
        }).set_index("date")

        # --- собираем прогноз Rt ---
        rt_median = np.median(rt_forecasts, axis=0)
        rt_low, rt_high = np.percentile(rt_forecasts, [2.5, 97.5], axis=0)
        df_rt_forecast = pd.DataFrame({
            "date": idx_future,
            "rt_median": rt_median,
            "rt_low95": rt_low,
            "rt_high95": rt_high
        }).set_index("date")

        # --- сохраняем прогноз ---
        df_forecast.to_csv(f"results/{self.region}_forecast.csv")

        # --- сохраняем все прогнозные Rt ---
        df_rt_forecast.to_csv(f"results/{self.region}_rt_forecast.csv")

        # --- сохраняем Rt_last ---
        rt_info = pd.DataFrame({
            "region": [self.region],
            "rt_last_mean": [rt_last_mean],
            "rt_last_low95": [rt_last_low],
            "rt_last_high95": [rt_last_high]
        })
        rt_info.to_csv(f"results/{self.region}_rt_last.csv", index=False)

        # --- строим графики ---
        self._plot_results(df_forecast)

        print(f"✅ Прогноз сохранён в results/{self.region}_forecast.csv")
        print(f"✅ Rt_last сохранён в results/{self.region}_rt_last.csv")
        print(f"✅ Rt прогноз сохранён в results/{self.region}_rt_forecast.csv")

        return df_forecast

    # === визуализация ===
    def _plot_results(self, df_forecast):
        import numpy as np
        import matplotlib.pyplot as plt
        import arviz as az
        import os

        obs = self.observed
        idata = self.idata
        region = self.region

        # Создаём папку results, если её нет
        os.makedirs("results", exist_ok=True)

        # === 1️⃣ График Rt ===
        r_t_mean = idata.posterior["r_t"].mean(dim=("chain", "draw")).values
        hdi_rt = az.hdi(idata.posterior["r_t"], hdi_prob=0.94)
        x_rt = np.arange(len(r_t_mean))

        # HDI Rt — в нужной форме
        hdi_rt_values = hdi_rt.to_array().values.squeeze().T
        hdi_rt_values = hdi_rt_values[:len(x_rt)]

        plt.figure(figsize=(10, 5))
        az.plot_hdi(x_rt, hdi_rt_values, color="lightblue")
        plt.plot(x_rt, r_t_mean, color="blue", label="Среднее Rt")
        plt.axhline(1, color="red", linestyle="--", label="Порог Rt=1")
        plt.title(f"R_t - {region}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{region}_R_t.png")
        plt.close()

        # === 2️⃣ График Infections ===
        infections_mean = idata.posterior["infections"].mean(dim=("chain", "draw")).values
        hdi_inf = az.hdi(idata.posterior["infections"], hdi_prob=0.94)
        x_inf = np.arange(len(infections_mean))

        hdi_inf_values = hdi_inf.to_array().values.squeeze().T
        hdi_inf_values = hdi_inf_values[:len(x_inf)]

        plt.figure(figsize=(10, 5))
        az.plot_hdi(x_inf, hdi_inf_values, color="orange")
        plt.plot(x_inf, infections_mean, color="darkorange", label="Среднее Infections")
        plt.title(f"Infections - {region}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{region}_infections.png")
        plt.close()

        # === 3️⃣ График прогноза (Positive) ===
        plt.figure(figsize=(10, 5))
        plt.plot(obs.index, obs["positive"], label="Наблюдаемые случаи", color="black")
        plt.plot(df_forecast.index, df_forecast["median"], label="Прогноз (медиана)", color="blue")
        plt.fill_between(
            df_forecast.index,
            df_forecast["low95"],
            df_forecast["high95"],
            color="blue",
            alpha=0.2,
            label="95% ДИ"
        )
        plt.title(f"Прогноз случаев - {region}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{region}_forecast.png")
        plt.close()

        print(f"📊 Все графики сохранены в папку results/")



# =============================================================================
# ШАГ 6: Запуск анализа для всех стран
# =============================================================================



import pickle
import pandas as pd

countries = ["Italy", "Germany", "France"]
for region in countries:


    # # Загружаем ваши данные
    df = load_country_data(region)

    model = CompiledCovidModel(region=region, observed=df)
    model.build_model()
    model.sample_fast(draws=500, tune=500)
    forecast_df = model.forecast(days=13)

    # Сохраняем прогноз
    forecast_df.to_csv(f"{region}_forecast.csv", index=False)
    print(f"📈 Прогноз сохранён в '{region}_forecast.csv'")

    # # Строим и сохраняем графики
    # model.plot_rt(save=True)
    # model.plot_cases(save=True)
    model._plot_results(forecast_df)
    print("✅ Все графики и прогноз успешно сохранены!")

    print(forecast_df.head())

    
# # Загружаем обученную модель
# with open(rf"C:\Data\Visual Studio\ВУЗ\Теорвер\results\{region}.pkl", "rb") as f:
#     model.idata = pickle.load(f)

# print("✅ Модель успешно загружена!")

# # Прогнозируем на 13 дней вперёд
# future_days = 13
# forecast_df = model.forecast(days=future_days)

# # Сохраняем прогноз
# forecast_df.to_csv("Russia_forecast.csv", index=False)
# print("📈 Прогноз сохранён в 'Russia_forecast.csv'")

# # # Строим и сохраняем графики
# model._plot_results(forecast_df)
# print("✅ Все графики и прогноз успешно сохранены!")




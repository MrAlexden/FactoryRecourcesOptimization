import numpy as np

def calculate_raw_material_costs(
    target_boxes: int,          # Целевой объём продукции (коробок)
    material_per_box: dict,     # Нормы расхода сырья на 1 коробку (кг)
    base_prices: dict,          # Базовые цены на сырьё (руб/кг)
    price_volatility: dict,     # Волатильность цен (% от цены)
    defect_rate: float = 0.05,  # Доля брака (5%)
    delivery_risk: float = 0.1, # Вероятность задержки поставки (10%)
    safety_stock_days: int = 7, # Страховой запас (дней)
    n_simulations: int = 1000   # Число сценариев Монте-Карло
) -> dict:
    """
    Возвращает:
    {
        "expected_cost": средние затраты (руб),
        "min_cost": минимальные затраты (руб),
        "max_cost": максимальные затраты (руб),
        "risk_above_budget": вероятность превышения базового бюджета (%),
        "safety_stock": страховой запас (кг)
    }
    """
    # 1. Расчёт общего объёма сырья с учётом брака
    adjusted_boxes = target_boxes * (1 + defect_rate)
    required_materials = {
        material: adjusted_boxes * amount 
        for material, amount in material_per_box.items()
    }
    
    # 2. Расчёт страхового запаса (на случай задержки)
    daily_usage = {
        material: amount / 30 
        for material, amount in required_materials.items()
    }
    safety_stock = {
        material: daily_usage[material] * safety_stock_days
        for material in daily_usage
    }
    
    # 3. Моделирование цен через Монте-Карло
    simulated_costs = np.zeros(n_simulations)
    for i in range(n_simulations):
        total_cost = 0
        for material in base_prices:
            # Генерация случайной цены (нормальное распределение)
            price = np.random.normal(
                loc=base_prices[material],
                scale=base_prices[material] * price_volatility[material]
            )
            # Учёт задержки поставки (увеличиваем объём закупки на страховой запас)
            if np.random.rand() < delivery_risk:
                amount = required_materials[material] + safety_stock[material]
            else:
                amount = required_materials[material]
            total_cost += price * amount
        simulated_costs[i] = total_cost
    
    # 4. Бюджет без рисков
    base_budget = sum(
        base_prices[material] * required_materials[material] 
        for material in base_prices
    )
    
    # 5. Анализ результатов
    return {
        "expected_cost": int(np.mean(simulated_costs)),
        "min_cost": int(np.min(simulated_costs)),
        "max_cost": int(np.max(simulated_costs)),
        "risk_above_budget": round(100 * np.mean(simulated_costs > base_budget), 1),
        "safety_stock": safety_stock
    }

def calculate_production_costs(
    target_boxes: int,                  # Целевой объём продукции (коробок)
    # Параметры для переменных затрат
    energy_per_box: float,              # кВт·ч на 1 коробку
    maintenance_per_box: float,         # ₽ на обслуживание на 1 коробку
    # Параметры для постоянных затрат
    rent: float,                        # Аренда помещений (₽/мес)
    utilities: float,                   # Коммунальные услуги (₽/мес)
    equipment_depreciation: float,      # Амортизация (₽/мес)
    certification: float,               # Сертификация (₽/мес)
    internal_logistics: float,          # Логистика внутри производства (₽/мес)
    it_infrastructure: float,           # IT-инфраструктура (₽/мес)
    waste_disposal: float,              # Утилизация отходов (₽/мес)
    production_tax: float,              # Налоги на производство (₽/мес)
    equipment_insurance: float,         # Страховка оборудования (₽/мес)
    # Параметры для рисков
    energy_price_mean: float = 8.0,     # Средняя цена за кВт·ч (₽)
    energy_price_std: float = 0.5,      # Волатильность цены энергии
    equipment_failure_rate: float = 0.05, # Вероятность поломки оборудования
    failure_extra_cost: float = 100000,  # Доп. затраты при поломке (₽)
    n_simulations: int = 1000           # Число сценариев Монте-Карло
) -> dict:
    """
    Возвращает словарь с расчётами:
    {
        "total_cost": средние затраты (₽),
        "min_cost": минимальные затраты (₽),
        "max_cost": максимальные затраты (₽),
        "failure_risk": вероятность превышения бюджета из-за поломки (%),
        "cost_breakdown": разбивка затрат по категориям
    }
    """
    
    # 1. Постоянные затраты (не зависят от объёма производства)
    fixed_costs = (
        rent + utilities + equipment_depreciation +
        certification + internal_logistics + it_infrastructure +
        waste_disposal + production_tax + equipment_insurance
    )
    
    # 2. Моделирование методом Монте-Карло
    simulated_costs = []
    for _ in range(n_simulations):
        # Генерация случайной цены на энергию
        energy_price = np.random.normal(energy_price_mean, energy_price_std)
        energy_cost = energy_per_box * target_boxes * energy_price
        
        # Учёт риска поломки оборудования
        if np.random.rand() < equipment_failure_rate:
            extra_cost = failure_extra_cost
        else:
            extra_cost = 0
        
        # Итоговые затраты для сценария
        total_cost = (
            energy_cost +
            (maintenance_per_box * target_boxes) +
            fixed_costs +
            extra_cost
        )
        simulated_costs.append(total_cost)
    
    # 3. Анализ результатов
    cost_breakdown = {
        "energy": energy_per_box * target_boxes * energy_price_mean,
        "maintenance": maintenance_per_box * target_boxes,
        "rent": rent,
        "utilities": utilities,
        "depreciation": equipment_depreciation,
        "certification": certification,
        "logistics": internal_logistics,
        "it": it_infrastructure,
        "waste": waste_disposal,
        "tax": production_tax,
        "insurance": equipment_insurance,
        "failure_risk_cost": failure_extra_cost * equipment_failure_rate
    }
    
    return {
        "total_cost": int(np.mean(simulated_costs)),
        "min_cost": int(np.min(simulated_costs)),
        "max_cost": int(np.max(simulated_costs)),
        "failure_risk": round(equipment_failure_rate * 100, 1),
        "cost_breakdown": {k: int(v) for k, v in cost_breakdown.items()}
    }

def calculate_storage_costs(
    # Основные параметры склада
    storage_volume: float,            # Общий объем склада (м³)
    used_volume: float,               # Используемый объем (м³)
    
    # Постоянные затраты
    rent_per_month: float,            # Аренда склада (₽/мес)
    security_systems_cost: float,      # Охрана и безопасность (₽/мес)
    wms_cost: float,                  # Система управления запасами (₽/мес)
    utilities_cost: float,             # Коммунальные услуги (₽/мес)
    insurance_rate: float,            # Ставка страховки (% от стоимости запасов)
    inventory_value: float,           # Стоимость хранимых запасов (₽)
    depreciation_cost: float,         # Амортизация оборудования (₽/мес)
    
    # Переменные затраты
    storage_cost_per_m3: float,       # Затраты на место (₽/м³)
    internal_logistics_cost: float,    # Логистика внутри склада (₽/мес)
    spoilage_rate: float,             # Норма потерь от порчи (% от стоимости)
    
    # Параметры рисков
    rent_volatility: float = 0.1,     # Волатильность цены аренды (%)
    spoilage_risk: float = 0.05,      # Вероятность повышенной порчи
    security_breach_risk: float = 0.02, # Вероятность кражи/повреждения
    n_simulations: int = 10000        # Количество симуляций Монте-Карло
) -> dict:
    """
    Рассчитывает затраты на хранение с учетом рисков методом Монте-Карло.
    
    Возвращает словарь с результатами:
    {
        "base_cost": базовые затраты без учета рисков,
        "avg_total_cost": средние затраты с учетом рисков,
        "min_cost": минимальные затраты,
        "max_cost": максимальные затраты,
        "risk_breakdown": разбивка рисковых затрат,
        "cost_breakdown": детализация затрат
    }
    """
    # Расчет базовых затрат
    base_fixed_costs = (
        rent_per_month +
        security_systems_cost +
        wms_cost +
        utilities_cost +
        (inventory_value * insurance_rate / 12) +  # Месячная страховка
        depreciation_cost
    )
    
    base_variable_costs = (
        (used_volume * storage_cost_per_m3) +
        internal_logistics_cost +
        (inventory_value * spoilage_rate)
    )
    
    base_cost = base_fixed_costs + base_variable_costs
    
    # Моделирование методом Монте-Карло
    simulated_costs = []
    risk_breakdown = {
        "rent_increase": 0,
        "extra_spoilage": 0,
        "security_incidents": 0
    }
    
    for _ in range(n_simulations):
        additional_costs = 0
        
        # Риск роста арендной платы
        if np.random.rand() < 0.1:  # 10% вероятность роста
            rent_increase = rent_per_month * rent_volatility
            additional_costs += rent_increase
            risk_breakdown["rent_increase"] += rent_increase / n_simulations
        
        # Риск повышенной порчи
        if np.random.rand() < spoilage_risk:
            extra_spoilage = inventory_value * spoilage_rate * 2  # Вдвое больше
            additional_costs += extra_spoilage
            risk_breakdown["extra_spoilage"] += extra_spoilage / n_simulations
        
        # Риск краж/повреждений
        if np.random.rand() < security_breach_risk:
            security_loss = inventory_value * 0.01  # Потеря 1% запасов
            additional_costs += security_loss
            risk_breakdown["security_incidents"] += security_loss / n_simulations
        
        simulated_costs.append(base_cost + additional_costs)
    
    # Анализ результатов
    avg_total_cost = np.mean(simulated_costs)
    min_cost = np.min(simulated_costs)
    max_cost = np.max(simulated_costs)
    
    # Детализация затрат
    cost_breakdown = {
        "fixed_costs": {
            "rent": rent_per_month,
            "security": security_systems_cost,
            "wms": wms_cost,
            "utilities": utilities_cost,
            "insurance": inventory_value * insurance_rate / 12,
            "depreciation": depreciation_cost
        },
        "variable_costs": {
            "storage": used_volume * storage_cost_per_m3,
            "logistics": internal_logistics_cost,
            "spoilage": inventory_value * spoilage_rate
        },
        "risk_costs": {
            k: round(v, 2) for k, v in risk_breakdown.items()
        }
    }
    
    return {
        "base_cost": round(base_cost, 2),
        "avg_total_cost": round(avg_total_cost, 2),
        "min_cost": round(min_cost, 2),
        "max_cost": round(max_cost, 2),
        "risk_breakdown": risk_breakdown,
        "cost_breakdown": cost_breakdown
    }

def calculate_logistics_costs(
    # Входные параметры
    raw_material_volume: float,          # Объём сырья для перевозки (м³)
    finished_goods_volume: float,        # Объём готовой продукции (м³)
    distance_supplier: float,            # Расстояние до поставщика (км)
    distance_customer: float,            # Расстояние до клиента (км)
    # Параметры автопарка
    truck_capacity: float = 10.0,        # Грузоподъёмность 1 грузовика (м³)
    truck_cost_per_km: float = 50,       # Стоимость 1 км пробега (₽)
    truck_fixed_cost: float = 100000,    # Фиксированные затраты (страховка, зарплата водителя)
    # Параметры подрядчика
    contractor_cost_per_m3: float = 500, # Стоимость перевозки 1 м³ (₽)
    contractor_delay_risk: float = 0.1,  # Вероятность задержки подрядчика (10%)
    # Риски
    fuel_price_mean: float = 60,         # Средняя цена топлива (₽/литр)
    fuel_price_std: float = 5,           # Волатильность цены топлива
    damage_risk: float = 0.05,           # Вероятность повреждения груза (5%)
    damage_cost_per_m3: float = 1000,    # Убытки при повреждении 1 м³ (₽)
    n_simulations: int = 1000            # Число сценариев Монте-Карло
) -> dict:
    """
    Возвращает словарь с расчётами:
    {
        "total_cost": средние затраты (₽),
        "min_cost": минимальные затраты (₽),
        "max_cost": максимальные затраты (₽),
        "optimal_strategy": "аренда" или "подрядчик",
        "risk_breakdown": {
            "delay_risk": вероятность задержки (%),
            "damage_risk": вероятность повреждения (%)
        }
    }
    """
    # 1. Расчёт базовых параметров
    total_volume = raw_material_volume + finished_goods_volume
    trucks_needed = int(np.ceil(total_volume / truck_capacity))
    
    # 2. Моделирование методом Монте-Карло
    simulated_own_costs = []
    simulated_contractor_costs = []
    
    for _ in range(n_simulations):
        # Случайные факторы
        fuel_price = np.random.normal(fuel_price_mean, fuel_price_std)
        is_delayed = np.random.rand() < contractor_delay_risk
        is_damaged = np.random.rand() < damage_risk
        
        # Затраты при использовании своего автопарка
        own_cost = (
            (truck_cost_per_km * (distance_supplier + distance_customer) * trucks_needed) +
            (truck_fixed_cost * trucks_needed) +
            (fuel_price * 0.1 * (distance_supplier + distance_customer) * trucks_needed)  # 0.1 л/км
        )
        if is_damaged:
            own_cost += damage_cost_per_m3 * total_volume * 0.5  # Условно 50% груза повреждено
        
        # Затраты при использовании подрядчика
        contractor_cost = contractor_cost_per_m3 * total_volume
        if is_delayed:
            contractor_cost *= 1.2  # Штраф 20% за задержку
        if is_damaged:
            contractor_cost += damage_cost_per_m3 * total_volume * 0.3  # Подрядчик покрывает 70%
        
        simulated_own_costs.append(own_cost)
        simulated_contractor_costs.append(contractor_cost)
    
    # 3. Анализ результатов
    mean_own_cost = np.mean(simulated_own_costs)
    mean_contractor_cost = np.mean(simulated_contractor_costs)
    
    return {
        "total_cost": int(min(mean_own_cost, mean_contractor_cost)),
        "min_cost": int(min(np.min(simulated_own_costs), np.min(simulated_contractor_costs))),
        "max_cost": int(max(np.max(simulated_own_costs), np.max(simulated_contractor_costs))),
        "optimal_strategy": "аренда" if mean_own_cost < mean_contractor_cost else "подрядчик",
        "risk_breakdown": {
            "delay_risk": round(contractor_delay_risk * 100, 1),
            "damage_risk": round(damage_risk * 100, 1)
        },
        "cost_breakdown": {
            "аренда": {
                "транспорт": int(truck_cost_per_km * (distance_supplier + distance_customer) * trucks_needed),
                "фиксированные_затраты": truck_fixed_cost * trucks_needed,
                "топливо": int(fuel_price_mean * 0.1 * (distance_supplier + distance_customer) * trucks_needed)
            },
            "подрядчик": {
                "перевозка": int(contractor_cost_per_m3 * total_volume),
                "штрафы_за_задержку": int(contractor_cost_per_m3 * total_volume * 0.2 * contractor_delay_risk)
            }
        }
    }

def calculate_labor_vs_automation(
    target_boxes: int,                  # Целевой объём производства (коробок/мес)
    # Параметры рабочей силы
    workers_productivity: float = 100,  # Производительность 1 рабочего (коробок/мес)
    worker_salary: float = 50000,       # Зарплата рабочего (₽/мес)
    worker_tax_rate: float = 0.3,       # Налоги на ФОТ (30%)
    worker_training_cost: float = 20000,# Затраты на обучение (₽/работника/год)
    # Параметры роботизации
    robot_productivity: float = 500,    # Производительность 1 робота (коробок/мес)
    robot_cost: float = 1000000,        # Стоимость 1 робота (₽)
    robot_lifespan: int = 60,           # Срок службы (мес)
    robot_maintenance: float = 20000,   # Обслуживание (₽/робота/мес)
    robot_software_cost: float = 50000, # ПО (₽/мес на всех роботов)
    # Дополнительные параметры
    discount_rate: float = 0.12,        # Ставка дисконтирования (12% годовых)
    n_years: int = 5,                   # Горизонт планирования (лет)
    risk_adjustment: float = 0.1        # Надбавка на риски (10%)
) -> dict:
    """
    Возвращает словарь с расчётами:
    {
        "optimal_solution": "рабочая сила" или "роботизация",
        "labor_cost": NPV затрат на рабочую силу (₽),
        "automation_cost": NPV затрат на роботизацию (₽),
        "break_even_months": срок окупаемости роботизации (мес),
        "details": детализация затрат
    }
    """
    # Конвертация лет в месяцы
    n_months = n_years * 12
    
    # 1. Расчёт затрат на рабочую силу
    n_workers = int(np.ceil(target_boxes / workers_productivity))
    monthly_labor_cost = n_workers * worker_salary * (1 + worker_tax_rate)
    annual_training = (worker_training_cost * n_workers) / 12  # В месяц
    
    # NPV для рабочей силы
    labor_cost_npv = 0
    for month in range(1, n_months + 1):
        monthly_cost = monthly_labor_cost + annual_training
        labor_cost_npv += monthly_cost / ((1 + discount_rate/12) ** month)
    
    # 2. Расчёт затрат на роботизацию
    n_robots = int(np.ceil(target_boxes / robot_productivity))
    
    # Единовременные затраты
    initial_robot_cost = n_robots * robot_cost
    
    # Регулярные затраты
    monthly_robot_cost = (
        (robot_maintenance * n_robots) +
        robot_software_cost +
        (initial_robot_cost / robot_lifespan)  # Амортизация
    )
    
    # NPV для роботизации
    automation_cost_npv = initial_robot_cost
    for month in range(1, n_months + 1):
        automation_cost_npv += monthly_robot_cost / ((1 + discount_rate/12) ** month)
    
    # 3. Учёт рисков (надбавка 10%)
    automation_cost_npv *= (1 + risk_adjustment)
    
    # 4. Определение оптимального решения
    optimal_solution = "labor" if labor_cost_npv < automation_cost_npv else "automation"
    
    # 5. Расчёт срока окупаемости роботизации
    break_even_months = None
    if optimal_solution == "automation":
        cumulative_savings = 0
        monthly_savings = monthly_labor_cost - monthly_robot_cost
        for month in range(1, n_months + 1):
            cumulative_savings += monthly_savings / ((1 + discount_rate/12) ** month)
            if cumulative_savings >= initial_robot_cost:
                break_even_months = month
                break
    
    return {
        "optimal_solution": optimal_solution,
        "labor_cost": int(labor_cost_npv),
        "automation_cost": int(automation_cost_npv),
        "break_even_months": break_even_months,
        "details": {
            "рабочая сила": {
                "количество_работников": n_workers,
                "ежемесячные_затраты": int(monthly_labor_cost + annual_training),
                "npc_5_лет": int(labor_cost_npv)
            },
            "роботизация": {
                "количество_роботов": n_robots,
                "начальные_затраты": initial_robot_cost,
                "ежемесячные_затраты": int(monthly_robot_cost),
                "npc_5_лет": int(automation_cost_npv)
            }
        }
    }

if __name__ == "__main__":
    # Параметры для производства конструктора (аналог LEGO)
    params = {
        "target_boxes": 10000,
        "material_per_box": {"plastic": 2.0, "dye": 0.5, "packaging": 0.1},
        "base_prices": {"plastic": 100, "dye": 200, "packaging": 50},
        "price_volatility": {"plastic": 0.15, "dye": 0.10, "packaging": 0.05},
        "defect_rate": 0.05,
        "delivery_risk": 0.1,
        "safety_stock_days": 7
    }
    
    result = calculate_raw_material_costs(**params)
    print("Результаты прогноза затрат на сырьё:")
    for key, value in result.items():
        if key != "safety_stock":
            print(f"{key.replace('_', ' ').title()}: {value:,} ₽" if "cost" in key else f"{key.replace('_', ' ').title()}: {value}%")
        else:
            print("\nСтраховой запас (кг):")
            for material, amount in value.items():
                print(f"  {material}: {amount:.1f}")
    print("\n")

    # Параметры для производства конструкторов (аналог LEGO)
    params = {
        "target_boxes": 10000,
        "energy_per_box": 2.5,          # кВт·ч на коробку
        "maintenance_per_box": 30,      # ₽ на коробку
        "rent": 500000,                 # Аренда помещений (₽/мес)
        "utilities": 200000,            # Коммунальные услуги (₽/мес)
        "equipment_depreciation": 250000, # Амортизация (₽/мес)
        "certification": 50000,         # Сертификация (₽/мес)
        "internal_logistics": 100000,    # Логистика внутри производства (₽/мес)
        "it_infrastructure": 150000,     # IT (₽/мес)
        "waste_disposal": 30000,         # Утилизация (₽/мес)
        "production_tax": 50000,         # Налоги (₽/мес)
        "equipment_insurance": 100000,   # Страховка (₽/мес)
        "energy_price_mean": 8.0,        # Средняя цена за кВт·ч (₽/мес)
        "energy_price_std": 0.8,         # Волатильность цены
        "equipment_failure_rate": 0.05,  # 5% риск поломки
        "failure_extra_cost": 200000     # Затраты на ремонт
    }
    
    results = calculate_production_costs(**params)
    
    print("=== Прогноз производственных издержек ===")
    print(f"Средние затраты: {results['total_cost']:,} ₽")
    print(f"Минимальные затраты: {results['min_cost']:,} ₽")
    print(f"Максимальные затраты: {results['max_cost']:,} ₽")
    print(f"Риск поломки оборудования: {results['failure_risk']}%")
    
    print("\n=== Детализация затрат ===")
    for category, cost in results["cost_breakdown"].items():
        print(f"{category.replace('_', ' ').title():<25}: {cost:,} ₽")

    # Параметры для склада конструкторов (аналог LEGO)
    params = {
        "storage_volume": 1000,
        "used_volume": 800,
        "rent_per_month": 500000,
        "security_systems_cost": 30000,
        "wms_cost": 15000,
        "utilities_cost": 20000,
        "insurance_rate": 0.012,  # 1.2% в год
        "inventory_value": 10000000,  # 10 млн ₽
        "depreciation_cost": 25000,
        "storage_cost_per_m3": 50,
        "internal_logistics_cost": 50000,
        "spoilage_rate": 0.01,  # 1%
        "rent_volatility": 0.15,
        "spoilage_risk": 0.05,
        "security_breach_risk": 0.02
    }
    
    results = calculate_storage_costs(**params)
    
    print("=== Прогноз затрат на хранение ===")
    print(f"Базовые затраты: {results['base_cost']:,.2f} ₽")
    print(f"Средние затраты с учетом рисков: {results['avg_total_cost']:,.2f} ₽")
    print(f"Минимальные затраты: {results['min_cost']:,.2f} ₽")
    print(f"Максимальные затраты: {results['max_cost']:,.2f} ₽")
    
    print("\n=== Детализация постоянных затрат ===")
    for category, cost in results["cost_breakdown"]["fixed_costs"].items():
        print(f"{category.title():<15}: {cost:,.2f} ₽")
    
    print("\n=== Детализация переменных затрат ===")
    for category, cost in results["cost_breakdown"]["variable_costs"].items():
        print(f"{category.title():<15}: {cost:,.2f} ₽")
    
    print("\n=== Рисковые надбавки ===")
    for risk, cost in results["cost_breakdown"]["risk_costs"].items():
        print(f"{risk.replace('_', ' ').title():<25}: {cost:,.2f} ₽")

    # Параметры для производства конструкторов (аналог LEGO)
    params = {
        "raw_material_volume": 200,      # м³ сырья
        "finished_goods_volume": 150,    # м³ готовой продукции
        "distance_supplier": 300,        # км до поставщика
        "distance_customer": 200,        # км до клиента
        "truck_capacity": 12,            # м³
        "truck_cost_per_km": 45,         # ₽
        "truck_fixed_cost": 80000,       # ₽/мес за 1 грузовик
        "contractor_cost_per_m3": 550,   # ₽
        "contractor_delay_risk": 0.15    # 15%
    }
    
    results = calculate_logistics_costs(**params)
    
    print("=== Прогноз логистических затрат ===")
    print(f"Оптимальная стратегия: {results['optimal_strategy'].upper()}")
    print(f"Средние затраты: {results['total_cost']:,} ₽")
    print(f"Минимальные затраты: {results['min_cost']:,} ₽")
    print(f"Максимальные затраты: {results['max_cost']:,} ₽")
    print(f"Риск задержки: {results['risk_breakdown']['delay_risk']}%")
    print(f"Риск повреждения груза: {results['risk_breakdown']['damage_risk']}%")
    
    print("\n=== Детализация затрат ===")
    for strategy, costs in results["cost_breakdown"].items():
        print(f"\n{strategy.upper()}:")
        for item, cost in costs.items():
            print(f"  {item.replace('_', ' ')}: {cost:,} ₽")

    # Параметры для производства конструкторов (аналог LEGO)
    params = {
        "target_boxes": 10000,
        "workers_productivity": 120,
        "worker_salary": 60000,
        "robot_productivity": 600,
        "robot_cost": 1200000,
        "robot_lifespan": 84,
        "n_years": 5
    }
    
    results = calculate_labor_vs_automation(**params)
    
    print("=== Сравнение рабочей силы и роботизации ===")
    print(f"Оптимальное решение: {results['optimal_solution'].upper()}")
    print(f"NPV затрат (5 лет) на рабочую силу: {results['labor_cost']:,} ₽")
    print(f"NPV затрат (5 лет) на роботизацию: {results['automation_cost']:,} ₽")
    
    if results['optimal_solution'] == "роботизация":
        print(f"Срок окупаемости: {results['break_even_months']} месяцев")
    
    print("\n=== Детализация ===")
    for solution, data in results["details"].items():
        print(f"\n{solution.upper()}:")
        for key, value in data.items():
            print(f"  {key.replace('_', ' ')}: {value:,}" + (" ₽" if "затраты" in key else ""))
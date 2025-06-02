import numpy as np
from scipy.optimize import minimize
import Costs
from Costs import calculate_raw_material_costs
from Costs import calculate_production_costs
from Costs import calculate_storage_costs
from Costs import calculate_logistics_costs
from Costs import calculate_labor_vs_automation

def optimize_business_costs(
    target_boxes: int,                  # Целевой объем производства
    current_inventory: dict,            # Текущие запасы {'сырье': кг, 'товар': коробки}
    material_per_box: dict,             # Нормы расхода сырья
    base_prices: dict,                  # Базовые цены на сырье
    price_volatility: dict,             # Волатильность цен
    defect_rate: float,                 # Доля брака (5%)
    delivery_risk: float,               # Вероятность задержки поставки (10%)
    safety_stock_days: int,             # Страховой запас (дней)
    n_simulations: int,                 # Число сценариев Монте-Карло
    production_params: dict,            # Параметры производства
    storage_params: dict,               # Параметры хранения
    logistics_params: dict,             # Параметры логистики
    labor_params: dict,                 # Параметры труда/автоматизации
    n_months: int = 12,                 # Горизонт планирования (месяцев)
    risk_tolerance: float = 0.1,        # Допустимый уровень риска (0-1)
    budget_constraint: float = None     # Ограничение бюджета (руб)
) -> dict:
    """
    Оптимизирует бизнес-затраты с учетом рисков и ограничений.
    
    Возвращает словарь с оптимальным планом и прогнозами:
    {
        "optimal_production": оптимальный объем производства по месяцам,
        "raw_material_orders": закупки сырья по месяцам,
        "total_cost": общие затраты,
        "cost_breakdown": разбивка затрат по категориям,
        "risk_analysis": анализ рисков,
        "inventory_projection": прогноз запасов
    }
    """
    # 1. Инициализация переменных для оптимизации
    # x[0:n_months] - производство по месяцам
    # x[n_months:2*n_months] - закупка сырья по месяцам
    initial_guess = np.concatenate([
        np.full(n_months, target_boxes),  # Начальное предположение по производству
        np.full(n_months, target_boxes * sum(material_per_box.values()) / n_months)  # Закупки сырья
    ])
    
    # 2. Ограничения
    constraints = [
        # Производство не может быть отрицательным
        {'type': 'ineq', 'fun': lambda x: x[0:n_months]},
        # Закупки сырья не могут быть отрицательными
        {'type': 'ineq', 'fun': lambda x: x[n_months:2*n_months]},
    ]
    
    if budget_constraint:
        # Ограничение по бюджету
        constraints.append({
            'type': 'ineq', 
            'fun': lambda x: budget_constraint - calculate_total_cost(
                x[0:n_months], x[n_months:2*n_months], current_inventory,
                material_per_box, base_prices, price_volatility,
                defect_rate, delivery_risk, safety_stock_days, n_simulations,
                production_params, storage_params, logistics_params, labor_params
            )['total_cost']
        })
    
    # 3. Функция цели (минимизация затрат с учетом рисков)
    def objective(x):
        production = x[0:n_months]
        raw_material_orders = x[n_months:2*n_months]
        
        # Рассчитываем общие затраты
        cost_result = calculate_total_cost(
            production, raw_material_orders, current_inventory,
            material_per_box, base_prices, price_volatility,
            defect_rate, delivery_risk, safety_stock_days, n_simulations,
            production_params, storage_params, logistics_params, labor_params
        )
        
        # Штраф за отклонение от целевого объема производства
        target_deviation_penalty = np.sum((production - target_boxes)**2) * 100
        
        # Штраф за превышение допустимого риска
        risk_penalty = max(0, cost_result['risk_metrics']['total_risk'] - risk_tolerance) * 1e6
        
        return cost_result['total_cost'] + target_deviation_penalty + risk_penalty
    
    # 4. Оптимизация
    bounds = [(0, None) for _ in range(2*n_months)]  # Все переменные >= 0
    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    # 5. Формирование результатов
    optimal_production = result.x[0:n_months]
    raw_material_orders = result.x[n_months:2*n_months]
    
    # Полный расчет затрат для оптимального плана
    final_cost = calculate_total_cost(
        optimal_production, raw_material_orders, current_inventory,
        material_per_box, base_prices, price_volatility,
        defect_rate, delivery_risk, safety_stock_days, n_simulations,
        production_params, storage_params, logistics_params, labor_params
    )
    
    # Прогноз запасов
    inventory_projection = project_inventory(
        optimal_production, raw_material_orders, current_inventory, material_per_box
    )
    
    return {
        "optimal_production": optimal_production.round().astype(int),
        "raw_material_orders": raw_material_orders.round().astype(int),
        "total_cost": final_cost['total_cost'],
        "cost_breakdown": final_cost['cost_breakdown'],
        "risk_analysis": final_cost['risk_metrics'],
        "inventory_projection": inventory_projection,
        "optimization_success": result.success,
        "message": result.message
    }

def calculate_total_cost(
    production_plan, raw_material_orders, current_inventory,
    material_per_box, base_prices, price_volatility,
    defect_rate, delivery_risk, safety_stock_days, n_simulations,
    production_params, storage_params, logistics_params, labor_params
):
    """Рассчитывает общие затраты для заданного плана производства и закупок."""
    total_months = len(production_plan)
    
    # Инициализация переменных
    total_raw_material_cost = 0
    total_production_cost = 0
    total_storage_cost = 0
    total_logistics_cost = 0
    total_labor_cost = 0
    
    risk_metrics = {
        'supply_risk': 0,
        'production_risk': 0,
        'demand_risk': 0,
        'total_risk': 0
    }
    
    # Расчет затрат по месяцам
    for month in range(total_months):
        # 1. Затраты на сырье
        raw_material_cost = calculate_raw_material_costs(
            target_boxes=raw_material_orders[month],
            material_per_box=material_per_box,
            base_prices=base_prices,
            price_volatility=price_volatility,
            defect_rate=defect_rate,
            delivery_risk=delivery_risk,
            safety_stock_days=safety_stock_days,
            n_simulations=n_simulations
        )
        total_raw_material_cost += raw_material_cost['expected_cost']
        risk_metrics['supply_risk'] += raw_material_cost['risk_above_budget'] / total_months
        
        # 2. Производственные затраты
        production_cost = calculate_production_costs(
            target_boxes=production_plan[month],
            **production_params,
            n_simulations=n_simulations
        )
        total_production_cost += production_cost['total_cost']
        risk_metrics['production_risk'] += production_cost['failure_risk'] / total_months
        
        # 3. Затраты на хранение
        storage_cost = calculate_storage_costs(
            inventory_value=(current_inventory['raw_material'] * base_prices['plastic'] + 
                           current_inventory['goods'] * production_params.get('product_value', 1000)),
            **storage_params,
            n_simulations=n_simulations
        )
        total_storage_cost += storage_cost['avg_total_cost']
        
        # 4. Логистические затраты
        logistics_cost = calculate_logistics_costs(
            raw_material_volume=raw_material_orders[month] * sum(material_per_box.values()) / 1000,
            finished_goods_volume=production_plan[month] * 0.01,  # примерный объем
            **logistics_params,
            n_simulations=n_simulations
        )
        total_logistics_cost += logistics_cost['total_cost']
        
        # Обновление запасов
        current_inventory['raw_material'] += raw_material_orders[month] - (production_plan[month] * sum(material_per_box.values()))
        current_inventory['goods'] += production_plan[month] - logistics_params.get('estimated_sales', production_plan[month]*0.9)
    
    # 5. Затраты на труд/автоматизацию (рассчитываем один раз для среднего объема)
    avg_production = np.mean(production_plan)
    labor_cost = calculate_labor_vs_automation(
        target_boxes=avg_production,
        **labor_params
    )
    optimal_solution = labor_cost['optimal_solution']
    total_labor_cost = labor_cost[f"{optimal_solution}_cost"] / total_months
    
    # Расчет рисков
    risk_metrics['demand_risk'] = max(0, (np.sum(production_plan) - (total_months * production_params.get('target_boxes', 10000))) / np.sum(production_plan))
    risk_metrics['total_risk'] = (risk_metrics['supply_risk'] + risk_metrics['production_risk'] + risk_metrics['demand_risk']) / 3
    
    return {
        'total_cost': total_raw_material_cost + total_production_cost + total_storage_cost + total_logistics_cost + total_labor_cost,
        'cost_breakdown': {
            'raw_materials': total_raw_material_cost,
            'production': total_production_cost,
            'storage': total_storage_cost,
            'logistics': total_logistics_cost,
            'labor': total_labor_cost
        },
        'risk_metrics': risk_metrics
    }

def project_inventory(production_plan, raw_material_orders, current_inventory, material_per_box):
    """Прогнозирует уровень запасов по месяцам."""
    inventory_projection = {
        'raw_material': [],
        'finished_goods': []
    }
    
    current_raw = current_inventory.get('raw_material', 0)
    current_goods = current_inventory.get('goods', 0)
    
    for month in range(len(production_plan)):
        # Обновляем запасы сырья
        materials_needed = production_plan[month] * sum(material_per_box.values())
        current_raw += raw_material_orders[month] - materials_needed
        current_raw = max(0, current_raw)  # Не может быть отрицательным
        
        # Обновляем запасы готовой продукции (предполагаем продажи = 90% производства)
        sales = production_plan[month] * 0.9
        current_goods += production_plan[month] - sales
        current_goods = max(0, current_goods)
        
        inventory_projection['raw_material'].append(current_raw)
        inventory_projection['finished_goods'].append(current_goods)
    
    return inventory_projection

# Пример использования
if __name__ == "__main__":
    # Параметры для оптимизации
    params = {
        "target_boxes": 10000,
        "current_inventory": {"raw_material": 5000, "goods": 2000},
        "material_per_box": {"plastic": 2.0, "dye": 0.5, "packaging": 0.1},
        "base_prices": {"plastic": 100, "dye": 200, "packaging": 50},
        "price_volatility": {"plastic": 0.15, "dye": 0.10, "packaging": 0.05},
        "defect_rate": 0.05,
        "delivery_risk": 0.1,
        "safety_stock_days": 7,
        "n_simulations": 10000,
        "production_params": {
            "energy_per_box": 2.5,
            "maintenance_per_box": 30,
            "rent": 500000,
            "utilities": 200000,
            "equipment_depreciation": 250000,
            "certification": 50000,
            "internal_logistics": 100000,
            "it_infrastructure": 150000,
            "waste_disposal": 30000,
            "production_tax": 50000,
            "equipment_insurance": 100000,
            "energy_price_mean": 8.0,
            "energy_price_std": 0.8,
            "equipment_failure_rate": 0.05,
            "failure_extra_cost": 200000,
            #"product_value": 1500  # Стоимость 1 коробки
        },
        "storage_params": {
            "storage_volume": 1000,
            "used_volume": 800,
            "rent_per_month": 500000,
            "security_systems_cost": 30000,
            "wms_cost": 15000,
            "utilities_cost": 20000,
            "insurance_rate": 0.012,
            "depreciation_cost": 25000,
            "storage_cost_per_m3": 50,
            "internal_logistics_cost": 50000,
            "spoilage_rate": 0.01,
            "rent_volatility": 0.15,
            "spoilage_risk": 0.05,
            "security_breach_risk": 0.02
        },
        "logistics_params": {
            "distance_supplier": 300,
            "distance_customer": 200,
            "truck_capacity": 12,
            "truck_cost_per_km": 45,
            "truck_fixed_cost": 80000,
            "contractor_cost_per_m3": 550,
            "contractor_delay_risk": 0.15,
            "fuel_price_mean": 60,
            "fuel_price_std": 5,
            "damage_risk": 0.05,
            "damage_cost_per_m3": 1000,
        },
        "labor_params": {
            "workers_productivity": 120,
            "worker_salary": 60000,
            "worker_tax_rate": 0.3,
            "worker_training_cost": 20000,
            "robot_productivity": 600,
            "robot_cost": 1200000,
            "robot_lifespan": 84,
            "robot_maintenance": 20000,
            "robot_software_cost": 50000,
            "discount_rate": 0.12,
            "n_years": 5,
            "risk_adjustment": 0.1
        },
        "n_months": 12,
        "risk_tolerance": 0.15,
        "budget_constraint": 15000000  # 15 млн руб в месяц
    }
    
    # Запуск оптимизации
    optimization_result = optimize_business_costs(**params)
    
    # Вывод результатов
    print("=== Результаты оптимизации бизнес-затрат ===")
    print(f"Общие затраты за период: {optimization_result['total_cost']:,.2f} ₽")
    print(f"Среднемесячные затраты: {optimization_result['total_cost'] / params['n_months']:,.2f} ₽")
    
    print("\nОптимальный план производства по месяцам:")
    for month, production in enumerate(optimization_result['optimal_production'], 1):
        print(f"Месяц {month}: {production} коробок")
    
    print("\nПлан закупок сырья по месяцам (кг):")
    for month, order in enumerate(optimization_result['raw_material_orders'], 1):
        print(f"Месяц {month}: {order:.0f} кг")
    
    print("\nРаспределение затрат:")
    for category, cost in optimization_result['cost_breakdown'].items():
        print(f"{category.title():<15}: {cost:,.2f} ₽ ({cost/optimization_result['total_cost']:.1%})")
    
    print("\nАнализ рисков:")
    for risk, value in optimization_result['risk_analysis'].items():
        print(f"{risk.replace('_', ' ').title():<20}: {value:.1%}")
    
    print("\nПрогноз запасов на конец периода:")
    print(f"Сырье: {optimization_result['inventory_projection']['raw_material'][-1]:.0f} кг")
    print(f"Готовая продукция: {optimization_result['inventory_projection']['finished_goods'][-1]:.0f} коробок")
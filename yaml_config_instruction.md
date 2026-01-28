# Подробная инструкция по составлению YAML конфигурационных файлов

## Содержание
1. [Основы синтаксиса YAML](#основы-синтаксиса-yaml)
2. [Структура конфигурационных файлов](#структура-конфигурационных-файлов)
3. [Тонкости и лучшие практики](#тонкости-и-лучшие-практики)
4. [Работа с YAML в Python](#работа-с-yaml-в-python)
5. [Работа с YAML в C++](#работа-с-yaml-в-c)

## Основы синтаксиса YAML

### 1. Ключ-значение
```yaml
# Простые значения
key: value
number: 42
float: 3.14
boolean: true
null_value: null

# Многострочные строки
description: |
  Это многострочная
  строка с сохранением
  переносов строк

# Сворачиваемые строки  
summary: >
  Это свернутая строка,
  переносы будут заменены
  пробелами
```

### 2. Списки
```yaml
# Простой список
fruits:
  - apple
  - orange
  - banana

# Встроенный синтаксис
colors: [red, green, blue]

# Список объектов
employees:
  - name: John
    age: 30
    position: developer
  - name: Jane
    age: 25
    position: designer
```

### 3. Словари (ассоциативные массивы)
```yaml
# Вложенные структуры
database:
  host: localhost
  port: 5432
  credentials:
    username: admin
    password: secret
  connections:
    max: 100
    timeout: 30s
```

### 4. Якоря и алиасы (повторное использование)
```yaml
# Определение якоря
defaults: &default_settings
  timeout: 30
  retries: 3
  log_level: INFO

# Использование алиаса
service_a:
  <<: *default_settings
  name: Service A

service_b:
  <<: *default_settings
  name: Service B
  timeout: 60  # Переопределение
```

## Структура конфигурационных файлов

### Пример конфига веб-приложения
```yaml
# config.yaml
version: 1.0

app:
  name: "My Application"
  environment: ${ENV:production}  # Переменная окружения с дефолтом
  debug: false
  log_level: INFO
  
server:
  host: 0.0.0.0
  port: 8080
  workers: 4
  ssl:
    enabled: true
    cert_path: /path/to/cert.pem
    key_path: /path/to/key.pem

database:
  primary:
    dialect: postgresql
    host: ${DB_HOST:localhost}
    port: ${DB_PORT:5432}
    name: ${DB_NAME:app_db}
    user: ${DB_USER:admin}
    password: ${DB_PASSWORD:}
    pool:
      min: 1
      max: 10
      idle_timeout: 300s
  
  redis:
    host: localhost
    port: 6379
    db: 0

features:
  enabled:
    - authentication
    - caching
    - api_rate_limiting
  settings:
    cache_ttl: 3600
    rate_limit: 100/hour

logging:
  handlers:
    console:
      enabled: true
      level: INFO
      format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    file:
      enabled: true
      path: /var/log/app.log
      max_size: 10MB
      backup_count: 5
```

### Пример конфига CI/CD
```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_IMAGE: registry.example.com/myapp:$CI_COMMIT_REF_SLUG

.test_template: &test_template
  stage: test
  image: python:3.9
  before_script:
    - pip install -r requirements.txt
  artifacts:
    paths:
      - test_reports/
    expire_in: 1 week

unit_tests:
  <<: *test_template
  script:
    - pytest tests/unit --junitxml=test_reports/unit.xml

integration_tests:
  <<: *test_template
  script:
    - pytest tests/integration --junitxml=test_reports/integration.xml

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $DOCKER_IMAGE .
    - docker push $DOCKER_IMAGE
  only:
    - master
    - tags

deploy_production:
  stage: deploy
  image: alpine:latest
  script:
    - apk add --no-cache curl
    - curl -X POST $DEPLOY_WEBHOOK
  environment:
    name: production
    url: https://app.example.com
  when: manual
  only:
    - master
```

## Тонкости и лучшие практики

### 1. **Чувствительность к отступам**
```yaml
# Правильно
parent:
  child:
    grandchild: value

# НЕПРАВИЛЬНО
parent:
child:  # Ошибка: одинаковый отступ
  grandchild: value
```

### 2. **Строки без кавычек и с кавычками**
```yaml
# Без кавычек (обычные строки)
path: /usr/local/bin
version: 1.0.0

# С кавычками (если есть спецсимволы)
special_string: "Значение: true"
another: 'Значение с : и # символами'

# Обязательно в кавычках если:
# - начинается с цифры: "123abc"
# - содержит :, #, {, }, [, ], |, >, -, ? и т.д.
# - совпадает с ключевым словом: "yes", "no", "true", "false", "null"
```

### 3. **Многострочные строки**
```yaml
# Сохранение переносов (|)
script: |
  echo "Line 1"
  echo "Line 2"
  echo "Line 3"

# Сворачивание переносов (>)
paragraph: >
  Это длинный абзац
  который будет свернут
  в одну строку с пробелами

# Сохранение последнего переноса (|+)
# Удаление последнего переноса (|-)
```

### 4. **Типы данных**
```yaml
# Явное указание типов
integer: !!int 42
float: !!float 3.14
string: !!str 123  # "123"
boolean: !!bool "true"  # true
null: !!null "null"  # null

# Временные метки
timestamp: !!timestamp 2023-12-31T23:59:59Z

# Бинарные данные
binary: !!binary |
  R0lGODlhDAAMAIQAAP//9/X
  17unp5WZmZgAAAOfn515eXv
  Pz7Y6OjuDg4J+fn5OTk6enp
  56enmleECcgggoBADs=
```

### 5. **Валидация схемы**
Всегда проверяйте конфигурацию на соответствие схеме. Пример схемы в JSON Schema:
```yaml
# config_schema.yaml
$schema: http://json-schema.org/draft-07/schema#
type: object
required:
  - app
  - server
properties:
  app:
    type: object
    properties:
      name:
        type: string
      environment:
        type: string
        enum: [development, staging, production]
      debug:
        type: boolean
      log_level:
        type: string
        enum: [DEBUG, INFO, WARNING, ERROR, CRITICAL]
```

## Работа с YAML в Python

### Основные библиотеки

#### 1. **PyYAML** (рекомендуемая)
```python
import yaml
import os
from typing import Dict, Any

# Безопасная загрузка (без исполнения кода)
def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Загрузка с поддержкой якорей
def load_config_full(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

# Сохранение конфига
def save_config(data: Dict[str, Any], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, 
                  allow_unicode=True, sort_keys=False)

# Пример с кастомным конструктором
class EnvVarLoader(yaml.SafeLoader):
    pass

def env_var_constructor(loader, node):
    value = loader.construct_scalar(node)
    var_name, default = value.split(':') if ':' in value else (value, '')
    return os.getenv(var_name, default)

EnvVarLoader.add_constructor('!env', env_var_constructor)

# Использование
config = """
app:
  host: !env HOST:localhost
  port: !env PORT:8080
"""

data = yaml.load(config, Loader=EnvVarLoader)
```

#### 2. **ruamel.yaml** (сохранение комментариев и порядка)
```python
from ruamel.yaml import YAML
from pathlib import Path

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)
yaml.preserve_quotes = True

# Загрузка с сохранением комментариев
config_path = Path('config.yaml')
data = yaml.load(config_path)

# Модификация и сохранение
data['app']['version'] = '2.0'
yaml.dump(data, config_path)
```

#### 3. **OmegaConf** (гибридные конфиги)
```python
from omegaconf import OmegaConf
import yaml

# Объединение конфигов
base_cfg = OmegaConf.load("base.yaml")
user_cfg = OmegaConf.load("user.yaml")
cli_cfg = OmegaConf.from_cli()  # Аргументы командной строки

# Слияние конфигураций
config = OmegaConf.merge(base_cfg, user_cfg, cli_cfg)

# Интерполяция значений
config = OmegaConf.create("""
server:
  host: localhost
  port: 8080
  url: http://${server.host}:${server.port}
""")

# Валидация
OmegaConf.set_struct(config, True)  # Запрещает добавление новых ключей
```

### Пример класса для работы с конфигом
```python
from dataclasses import dataclass, field
from typing import List, Optional
import yaml

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    name: str = "database"
    user: str = "admin"
    password: str = ""
    pool_size: int = 10

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4
    ssl_enabled: bool = False

@dataclass
class AppConfig:
    name: str = "MyApp"
    debug: bool = False
    log_level: str = "INFO"
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'AppConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Конвертация словарей в датаклассы
        if 'database' in data:
            data['database'] = DatabaseConfig(**data['database'])
        if 'server' in data:
            data['server'] = ServerConfig(**data['server'])
        
        return cls(**data)
    
    def to_yaml(self, path: str):
        data = {
            'name': self.name,
            'debug': self.debug,
            'log_level': self.log_level,
            'database': self.__dict__['database'].__dict__,
            'server': self.__dict__['server'].__dict__
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

# Использование
config = AppConfig.from_yaml('config.yaml')
print(config.database.host)
```

## Работа с YAML в C++

### Основные библиотеки

#### 1. **yaml-cpp** (наиболее популярная)
```cpp
#include <iostream>
#include "yaml-cpp/yaml.h"

// Чтение конфига
bool loadConfig(const std::string& filename) {
    try {
        YAML::Node config = YAML::LoadFile(filename);
        
        // Доступ к значениям
        std::string appName = config["app"]["name"].as<std::string>();
        int port = config["server"]["port"].as<int>();
        bool debug = config["app"]["debug"].as<bool>();
        
        // Работа со списками
        if (config["features"]["enabled"]) {
            for (const auto& feature : config["features"]["enabled"]) {
                std::cout << "Feature: " << feature.as<std::string>() << std::endl;
            }
        }
        
        // Проверка существования узла
        if (config["database"]["redis"]) {
            std::string redisHost = config["database"]["redis"]["host"].as<std::string>();
        }
        
        return true;
    } catch (const YAML::Exception& e) {
        std::cerr << "YAML Error: " << e.what() << std::endl;
        return false;
    }
}

// Запись конфига
void saveConfig(const std::string& filename) {
    YAML::Node config;
    
    // Создание структуры
    config["app"]["name"] = "MyApplication";
    config["app"]["version"] = "1.0.0";
    config["app"]["debug"] = false;
    
    config["server"]["host"] = "0.0.0.0";
    config["server"]["port"] = 8080;
    
    // Списки
    config["modules"].push_back("auth");
    config["modules"].push_back("api");
    config["modules"].push_back("database");
    
    // Вложенные структуры
    YAML::Node database;
    database["host"] = "localhost";
    database["port"] = 5432;
    config["database"] = database;
    
    // Сохранение в файл
    std::ofstream fout(filename);
    fout << config;
}

// Пример класса конфигурации
class AppConfig {
private:
    struct Database {
        std::string host;
        int port;
        std::string name;
        
        void fromYaml(const YAML::Node& node) {
            host = node["host"].as<std::string>();
            port = node["port"].as<int>();
            name = node["name"].as<std::string>();
        }
    };
    
    struct Server {
        std::string host;
        int port;
        int workers;
        
        void fromYaml(const YAML::Node& node) {
            host = node["host"].as<std::string>();
            port = node["port"].as<int>();
            workers = node["workers"].as<int>(4); // значение по умолчанию
        }
    };
    
    Database database_;
    Server server_;
    std::string name_;
    bool debug_;
    
public:
    bool loadFromFile(const std::string& path) {
        try {
            YAML::Node config = YAML::LoadFile(path);
            
            name_ = config["app"]["name"].as<std::string>();
            debug_ = config["app"]["debug"].as<bool>(false);
            
            if (config["database"]) {
                database_.fromYaml(config["database"]);
            }
            
            if (config["server"]) {
                server_.fromYaml(config["server"]);
            }
            
            return true;
        } catch (const YAML::Exception& e) {
            std::cerr << "Failed to load config: " << e.what() << std::endl;
            return false;
        }
    }
    
    // ... остальные методы
};
```

#### 2. **Modern C++ обертка для yaml-cpp**
```cpp
#include <optional>
#include <variant>
#include <vector>
#include <map>
#include "yaml-cpp/yaml.h"

class YamlConfig {
private:
    YAML::Node root_;
    
public:
    explicit YamlConfig(const std::string& filename) {
        root_ = YAML::LoadFile(filename);
    }
    
    explicit YamlConfig(const YAML::Node& node) : root_(node) {}
    
    // Шаблонные методы доступа
    template<typename T>
    std::optional<T> get(const std::string& key) const {
        try {
            return root_[key].as<T>();
        } catch (...) {
            return std::nullopt;
        }
    }
    
    template<typename T>
    T getOr(const std::string& key, const T& defaultValue) const {
        if (auto value = get<T>(key)) {
            return *value;
        }
        return defaultValue;
    }
    
    template<typename T>
    std::vector<T> getVector(const std::string& key) const {
        std::vector<T> result;
        if (root_[key] && root_[key].IsSequence()) {
            for (const auto& item : root_[key]) {
                result.push_back(item.as<T>());
            }
        }
        return result;
    }
    
    YamlConfig getSubConfig(const std::string& key) const {
        return YamlConfig(root_[key]);
    }
    
    bool has(const std::string& key) const {
        return root_[key];
    }
};

// Использование
YamlConfig config("settings.yaml");
auto host = config.get<std::string>("host").value_or("localhost");
auto port = config.getOr<int>("port", 8080);
auto features = config.getVector<std::string>("features.enabled");
```

### Сборка и зависимости

#### Для yaml-cpp (CMake):
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(MyApp)

# Поиск yaml-cpp
find_package(yaml-cpp REQUIRED)

add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE yaml-cpp)
```

#### Установка через пакетные менеджеры:
```bash
# Ubuntu/Debian
sudo apt-get install libyaml-cpp-dev

# macOS с Homebrew
brew install yaml-cpp

# vcpkg
vcpkg install yaml-cpp

# Conan
conan install yaml-cpp/0.7.0@
```

## Заключение

### Ключевые рекомендации:

1. **Всегда используйте safe_load/safe_dump** в Python для предотвращения инъекций кода
2. **Валидируйте конфигурацию** с помощью JSON Schema или pydantic
3. **Разделяйте конфигурацию** на базовую, окружение и локальные настройки
4. **Используйте переменные окружения** для секретных данных
5. **Сохраняйте порядок ключей** и комментарии при редактировании
6. **Предоставляйте значения по умолчанию**
7. **Документируйте все параметры** конфигурации

### Выбор библиотеки:

- **Python**: Для большинства задач используйте **PyYAML**, для сложных конфигов с интерполяцией - **OmegaConf**
- **C++**: **yaml-cpp** - стандартный выбор, хорошо поддерживается и документирован

YAML предоставляет отличный баланс между читаемостью и функциональностью, что делает его идеальным выбором для конфигурационных файлов в современных приложениях.
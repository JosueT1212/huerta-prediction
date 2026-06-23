create table if not exists phenology_observations (
  id                     bigserial   primary key,
  greenhouse_id          int         not null,
  week_date              date        not null,
  zona                   int         not null check (zona between 1 and 4),
  planta                 int         not null check (planta between 1 and 4),
  recorded_at            timestamptz not null default now(),
  racimos_puestos        float8,
  flores_racimo_abiertas float8,
  racimos_en_planta      float8,
  cantidad_tomates       float8,
  racimo_en_cosecha      float8,
  tomates_maduros        float8,
  diametro_fruto_cm      float8,
  crecimiento_planta_cm  float8
);

create unique index if not exists phenology_obs_unique
  on phenology_observations (greenhouse_id, week_date, zona, planta);

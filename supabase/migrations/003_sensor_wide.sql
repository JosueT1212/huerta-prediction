create table if not exists sensor_readings_wide (
  id            bigserial   primary key,
  greenhouse_id int         not null,
  fecha         date        not null,
  temp_prom_int float8,
  temp_min_int  float8,
  temp_max_int  float8,
  hr_prom_int   float8,
  co2_ppm       float8,
  riego_total   float8,
  ph_promedio   float8,
  ce_promedio   float8,
  temp_prom_ext float8,
  temp_max_ext  float8,
  temp_min_ext  float8,
  rad_sum       float8
);

create unique index if not exists sensor_wide_gh_date
  on sensor_readings_wide (greenhouse_id, fecha);

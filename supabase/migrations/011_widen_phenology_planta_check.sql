-- supabase/migrations/011_widen_phenology_planta_check.sql
--
-- 004_phenology_live.sql constrained `planta between 1 and 4`, which was
-- too narrow: real T18 fenologia data tracks up to 10 plants per zona
-- (Data/T_18/Monitoreo de Fenologia - Temporada 18.xlsx), so uploads were
-- failing the check constraint with an uncaught 500 on
-- POST /uploads/{inv}/fenologia (confirmed via Railway logs — postgrest
-- APIError 23514, "Failing row contains (... planta=5 ...)"). Drop the
-- arbitrary upper bound — plant count per zona is an agronomic choice,
-- not a fixed schema fact.

alter table phenology_observations
  drop constraint if exists phenology_observations_planta_check;

alter table phenology_observations
  add constraint phenology_observations_planta_check check (planta > 0);

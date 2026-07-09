-- supabase/migrations/006_submission_locks.sql
create table if not exists submission_locks (
  greenhouse_id      int         not null,
  form_type          text        not null check (form_type in ('sensores','fenologia','produccion')),
  last_submitted_at  timestamptz not null default now(),
  primary key (greenhouse_id, form_type)
);

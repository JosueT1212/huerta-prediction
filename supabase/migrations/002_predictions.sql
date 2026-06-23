create table if not exists predictions (
  id            bigserial   primary key,
  greenhouse_id int         not null,
  predicted_for date        not null,
  predicted_at  timestamptz not null default now(),
  kg_predicted  float8      not null,
  kg_actual     float8,
  model_version text        not null default 'cnn_rnn_v1'
);

create unique index if not exists predictions_gh_week
  on predictions (greenhouse_id, predicted_for);

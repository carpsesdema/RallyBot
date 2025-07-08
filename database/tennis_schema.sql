-- Tennis Intelligence Database Schema (PostgreSQL Version)

CREATE TABLE IF NOT EXISTS players (
    id SERIAL PRIMARY KEY,
    api_player_id VARCHAR(50) UNIQUE,
    name VARCHAR(200) NOT NULL,
    short_name VARCHAR(100),
    gender VARCHAR(10),
    country_code VARCHAR(3),
    country_name VARCHAR(100),
    date_of_birth DATE,
    turned_pro INTEGER,
    height_cm INTEGER,
    weight_kg INTEGER,
    plays VARCHAR(20),
    backhand VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS player_rankings (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    ranking_date DATE,
    atp_ranking INTEGER,
    wta_ranking INTEGER,
    ranking_points INTEGER,
    ranking_movement INTEGER,
    weeks_at_ranking INTEGER,
    previous_ranking INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_id, ranking_date)
);

CREATE TABLE IF NOT EXISTS player_statistics (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    stat_date DATE,
    surface VARCHAR(20),
    timeframe VARCHAR(20),
    matches_played INTEGER DEFAULT 0,
    matches_won INTEGER DEFAULT 0,
    matches_lost INTEGER DEFAULT 0,
    win_percentage DECIMAL(5,2),
    sets_won INTEGER DEFAULT 0,
    sets_lost INTEGER DEFAULT 0,
    straight_sets_wins INTEGER DEFAULT 0,
    three_set_wins INTEGER DEFAULT 0,
    five_set_wins INTEGER DEFAULT 0,
    aces_per_match DECIMAL(4,2),
    double_faults_per_match DECIMAL(4,2),
    first_serve_percentage DECIMAL(5,2),
    first_serve_points_won DECIMAL(5,2),
    second_serve_points_won DECIMAL(5,2),
    break_points_saved DECIMAL(5,2),
    service_games_won DECIMAL(5,2),
    first_return_points_won DECIMAL(5,2),
    second_return_points_won DECIMAL(5,2),
    break_points_converted DECIMAL(5,2),
    return_games_won DECIMAL(5,2),
    tiebreaks_won INTEGER DEFAULT 0,
    tiebreaks_played INTEGER DEFAULT 0,
    deciding_sets_won INTEGER DEFAULT 0,
    deciding_sets_played INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_id, stat_date, surface, timeframe)
);

CREATE TABLE IF NOT EXISTS head_to_head (
    id SERIAL PRIMARY KEY,
    player1_id INTEGER REFERENCES players(id),
    player2_id INTEGER REFERENCES players(id),
    total_matches INTEGER DEFAULT 0,
    player1_wins INTEGER DEFAULT 0,
    player2_wins INTEGER DEFAULT 0,
    clay_matches INTEGER DEFAULT 0,
    clay_player1_wins INTEGER DEFAULT 0,
    grass_matches INTEGER DEFAULT 0,
    grass_player1_wins INTEGER DEFAULT 0,
    hard_matches INTEGER DEFAULT 0,
    hard_player1_wins INTEGER DEFAULT 0,
    last_5_player1_wins INTEGER DEFAULT 0,
    last_match_date DATE,
    last_match_winner_id INTEGER REFERENCES players(id),
    grand_slam_matches INTEGER DEFAULT 0,
    grand_slam_player1_wins INTEGER DEFAULT 0,
    masters_matches INTEGER DEFAULT 0,
    masters_player1_wins INTEGER DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player1_id, player2_id)
);

CREATE TABLE IF NOT EXISTS matches (
    id SERIAL PRIMARY KEY,
    api_match_id VARCHAR(100) UNIQUE,
    match_date DATE,
    tournament_name VARCHAR(200),
    tournament_level VARCHAR(50),
    round_name VARCHAR(50),
    surface VARCHAR(20),
    court_name VARCHAR(100),
    player1_id INTEGER REFERENCES players(id),
    player2_id INTEGER REFERENCES players(id),
    winner_id INTEGER REFERENCES players(id),
    score_summary VARCHAR(100),
    sets_won_player1 INTEGER,
    sets_won_player2 INTEGER,
    games_won_player1 INTEGER,
    games_won_player2 INTEGER,
    match_duration_minutes INTEGER,
    player1_aces INTEGER,
    player2_aces INTEGER,
    player1_double_faults INTEGER,
    player2_double_faults INTEGER,
    player1_first_serve_pct DECIMAL(5,2),
    player2_first_serve_pct DECIMAL(5,2),
    player1_break_points_won INTEGER,
    player2_break_points_won INTEGER,
    player1_break_points_total INTEGER,
    player2_break_points_total INTEGER,
    player1_odds_open DECIMAL(6,2),
    player2_odds_open DECIMAL(6,2),
    player1_odds_close DECIMAL(6,2),
    player2_odds_close DECIMAL(6,2),
    total_games_line DECIMAL(4,1),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS player_injuries (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    injury_date DATE,
    injury_type VARCHAR(100),
    body_part VARCHAR(50),
    severity VARCHAR(20),
    expected_return_date DATE,
    actual_return_date DATE,
    tournaments_missed INTEGER DEFAULT 0,
    source VARCHAR(100),
    impact_on_ranking INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS betting_analysis (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    analysis_date DATE,
    times_favorite INTEGER DEFAULT 0,
    favorite_wins INTEGER DEFAULT 0,
    favorite_win_rate DECIMAL(5,2),
    times_underdog INTEGER DEFAULT 0,
    underdog_wins INTEGER DEFAULT 0,
    underdog_win_rate DECIMAL(5,2),
    roi_as_favorite DECIMAL(6,2),
    roi_as_underdog DECIMAL(6,2),
    avg_odds_when_favorite DECIMAL(6,2),
    avg_odds_when_underdog DECIMAL(6,2),
    clay_betting_record VARCHAR(20),
    grass_betting_record VARCHAR(20),
    hard_betting_record VARCHAR(20),
    last_10_matches_record VARCHAR(20),
    last_30_days_form VARCHAR(20),
    current_streak VARCHAR(20),
    closing_line_value DECIMAL(5,2),
    steam_moves_for INTEGER DEFAULT 0,
    steam_moves_against INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_id, analysis_date)
);

CREATE TABLE IF NOT EXISTS tournament_performance (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    tournament_name VARCHAR(200),
    year INTEGER,
    surface VARCHAR(20),
    rounds_reached INTEGER,
    prize_money INTEGER,
    ranking_points INTEGER,
    matches_won INTEGER,
    matches_lost INTEGER,
    best_win VARCHAR(200),
    worst_loss VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_id, tournament_name, year)
);

CREATE TABLE IF NOT EXISTS data_sources (
    id SERIAL PRIMARY KEY,
    source_name VARCHAR(100),
    endpoint VARCHAR(200),
    last_successful_call TIMESTAMPTZ,
    total_calls_today INTEGER DEFAULT 0,
    total_errors_today INTEGER DEFAULT 0,
    rate_limit_remaining INTEGER,
    api_key_status VARCHAR(20),
    response_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_players_name ON players(name);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_h2h_players ON head_to_head(player1_id, player2_id);
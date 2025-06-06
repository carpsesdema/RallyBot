-- Tennis Intelligence Database Schema
-- Designed for comprehensive tennis data storage and analysis

-- Core Players Table
CREATE TABLE players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    plays VARCHAR(20), -- left/right handed
    backhand VARCHAR(20), -- one/two handed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily Rankings Snapshots
CREATE TABLE player_rankings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER,
    ranking_date DATE,
    atp_ranking INTEGER,
    wta_ranking INTEGER,
    ranking_points INTEGER,
    ranking_movement INTEGER, -- change from previous week
    weeks_at_ranking INTEGER,
    previous_ranking INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES players(id),
    UNIQUE(player_id, ranking_date)
);

-- Comprehensive Player Statistics
CREATE TABLE player_statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER,
    stat_date DATE,
    surface VARCHAR(20), -- all, clay, grass, hard
    timeframe VARCHAR(20), -- ytd, last52weeks, career

    -- Match Statistics
    matches_played INTEGER DEFAULT 0,
    matches_won INTEGER DEFAULT 0,
    matches_lost INTEGER DEFAULT 0,
    win_percentage DECIMAL(5,2),

    -- Set Statistics
    sets_won INTEGER DEFAULT 0,
    sets_lost INTEGER DEFAULT 0,
    straight_sets_wins INTEGER DEFAULT 0,
    three_set_wins INTEGER DEFAULT 0,
    five_set_wins INTEGER DEFAULT 0,

    -- Serve Statistics
    aces_per_match DECIMAL(4,2),
    double_faults_per_match DECIMAL(4,2),
    first_serve_percentage DECIMAL(5,2),
    first_serve_points_won DECIMAL(5,2),
    second_serve_points_won DECIMAL(5,2),
    break_points_saved DECIMAL(5,2),
    service_games_won DECIMAL(5,2),

    -- Return Statistics
    first_return_points_won DECIMAL(5,2),
    second_return_points_won DECIMAL(5,2),
    break_points_converted DECIMAL(5,2),
    return_games_won DECIMAL(5,2),

    -- Under Pressure
    tiebreaks_won INTEGER DEFAULT 0,
    tiebreaks_played INTEGER DEFAULT 0,
    deciding_sets_won INTEGER DEFAULT 0,
    deciding_sets_played INTEGER DEFAULT 0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES players(id),
    UNIQUE(player_id, stat_date, surface, timeframe)
);

-- Head-to-Head Records
CREATE TABLE head_to_head (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player1_id INTEGER,
    player2_id INTEGER,

    -- Overall H2H
    total_matches INTEGER DEFAULT 0,
    player1_wins INTEGER DEFAULT 0,
    player2_wins INTEGER DEFAULT 0,

    -- Surface Breakdown
    clay_matches INTEGER DEFAULT 0,
    clay_player1_wins INTEGER DEFAULT 0,
    grass_matches INTEGER DEFAULT 0,
    grass_player1_wins INTEGER DEFAULT 0,
    hard_matches INTEGER DEFAULT 0,
    hard_player1_wins INTEGER DEFAULT 0,

    -- Recent Form (last 5 meetings)
    last_5_player1_wins INTEGER DEFAULT 0,
    last_match_date DATE,
    last_match_winner_id INTEGER,

    -- Tournament Level Breakdown
    grand_slam_matches INTEGER DEFAULT 0,
    grand_slam_player1_wins INTEGER DEFAULT 0,
    masters_matches INTEGER DEFAULT 0,
    masters_player1_wins INTEGER DEFAULT 0,

    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player1_id) REFERENCES players(id),
    FOREIGN KEY (player2_id) REFERENCES players(id),
    FOREIGN KEY (last_match_winner_id) REFERENCES players(id),
    UNIQUE(player1_id, player2_id)
);

-- Match Results (detailed match data)
CREATE TABLE matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    api_match_id VARCHAR(100) UNIQUE,

    -- Match Details
    match_date DATE,
    tournament_name VARCHAR(200),
    tournament_level VARCHAR(50), -- grand_slam, masters, atp500, etc.
    round_name VARCHAR(50),
    surface VARCHAR(20),
    court_name VARCHAR(100),

    -- Players
    player1_id INTEGER,
    player2_id INTEGER,
    winner_id INTEGER,

    -- Score Details
    score_summary VARCHAR(100), -- "6-4, 7-6, 6-2"
    sets_won_player1 INTEGER,
    sets_won_player2 INTEGER,
    games_won_player1 INTEGER,
    games_won_player2 INTEGER,
    match_duration_minutes INTEGER,

    -- Match Statistics
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

    -- Betting Data (when available)
    player1_odds_open DECIMAL(6,2),
    player2_odds_open DECIMAL(6,2),
    player1_odds_close DECIMAL(6,2),
    player2_odds_close DECIMAL(6,2),
    total_games_line DECIMAL(4,1),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player1_id) REFERENCES players(id),
    FOREIGN KEY (player2_id) REFERENCES players(id),
    FOREIGN KEY (winner_id) REFERENCES players(id)
);

-- Injury Tracking
CREATE TABLE player_injuries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER,
    injury_date DATE,
    injury_type VARCHAR(100),
    body_part VARCHAR(50),
    severity VARCHAR(20), -- minor, moderate, major
    expected_return_date DATE,
    actual_return_date DATE,
    tournaments_missed INTEGER DEFAULT 0,
    source VARCHAR(100), -- official, media, social
    impact_on_ranking INTEGER, -- ranking drop during injury

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES players(id)
);

-- Betting Performance Tracking
CREATE TABLE betting_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER,
    analysis_date DATE,

    -- Favorite/Underdog Performance
    times_favorite INTEGER DEFAULT 0,
    favorite_wins INTEGER DEFAULT 0,
    favorite_win_rate DECIMAL(5,2),

    times_underdog INTEGER DEFAULT 0,
    underdog_wins INTEGER DEFAULT 0,
    underdog_win_rate DECIMAL(5,2),

    -- Value Metrics
    roi_as_favorite DECIMAL(6,2),
    roi_as_underdog DECIMAL(6,2),
    avg_odds_when_favorite DECIMAL(6,2),
    avg_odds_when_underdog DECIMAL(6,2),

    -- Surface Betting Performance
    clay_betting_record VARCHAR(20), -- "15-8" format
    grass_betting_record VARCHAR(20),
    hard_betting_record VARCHAR(20),

    -- Recent Form Indicators
    last_10_matches_record VARCHAR(20),
    last_30_days_form VARCHAR(20),
    current_streak VARCHAR(20), -- "W5" or "L2"

    -- Market Efficiency Indicators
    closing_line_value DECIMAL(5,2), -- how often they beat closing odds
    steam_moves_for INTEGER DEFAULT 0, -- sharp money movements
    steam_moves_against INTEGER DEFAULT 0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES players(id),
    UNIQUE(player_id, analysis_date)
);

-- Tournament Performance
CREATE TABLE tournament_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER,
    tournament_name VARCHAR(200),
    year INTEGER,
    surface VARCHAR(20),

    -- Performance Details
    rounds_reached INTEGER, -- 1=first round, 7=champion
    prize_money INTEGER,
    ranking_points INTEGER,
    matches_won INTEGER,
    matches_lost INTEGER,

    -- Notable Achievements
    best_win VARCHAR(200), -- "Beat #3 Alcaraz in QF"
    worst_loss VARCHAR(200),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES players(id),
    UNIQUE(player_id, tournament_name, year)
);

-- API Data Sources Tracking
CREATE TABLE data_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name VARCHAR(100),
    endpoint VARCHAR(200),
    last_successful_call TIMESTAMP,
    total_calls_today INTEGER DEFAULT 0,
    total_errors_today INTEGER DEFAULT 0,
    rate_limit_remaining INTEGER,
    api_key_status VARCHAR(20), -- active, expired, limited
    response_time_ms INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for Performance
CREATE INDEX idx_players_name ON players(name);
CREATE INDEX idx_players_country ON players(country_code);
CREATE INDEX idx_rankings_date ON player_rankings(ranking_date);
CREATE INDEX idx_rankings_player_date ON player_rankings(player_id, ranking_date);
CREATE INDEX idx_stats_player_surface ON player_statistics(player_id, surface);
CREATE INDEX idx_matches_date ON matches(match_date);
CREATE INDEX idx_matches_players ON matches(player1_id, player2_id);
CREATE INDEX idx_h2h_players ON head_to_head(player1_id, player2_id);
CREATE INDEX idx_injuries_player ON player_injuries(player_id, injury_date);
CREATE INDEX idx_betting_player_date ON betting_analysis(player_id, analysis_date);

-- Views for Common Queries
CREATE VIEW current_rankings AS
SELECT
    p.name,
    p.country_code,
    pr.atp_ranking,
    pr.wta_ranking,
    pr.ranking_points,
    pr.ranking_movement,
    pr.ranking_date
FROM players p
JOIN player_rankings pr ON p.id = pr.player_id
WHERE pr.ranking_date = (
    SELECT MAX(ranking_date)
    FROM player_rankings pr2
    WHERE pr2.player_id = pr.player_id
);

CREATE VIEW top_betting_performers AS
SELECT
    p.name,
    ba.favorite_win_rate,
    ba.underdog_win_rate,
    ba.roi_as_favorite,
    ba.roi_as_underdog,
    ba.last_10_matches_record,
    ba.analysis_date
FROM players p
JOIN betting_analysis ba ON p.id = ba.player_id
WHERE ba.analysis_date >= date('now', '-30 days')
ORDER BY ba.roi_as_underdog DESC;

-- Triggers for Data Integrity
CREATE TRIGGER update_player_timestamp
    AFTER UPDATE ON players
BEGIN
    UPDATE players SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_h2h_timestamp
    AFTER UPDATE ON head_to_head
BEGIN
    UPDATE head_to_head SET last_updated = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
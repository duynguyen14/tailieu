/*
Store OCR text and immutable source coordinates used to ground LLM outputs.
The script is idempotent and never deletes existing OCR runs or tokens.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

IF OBJECT_ID(N'dbo.annotation_pages', N'U') IS NULL
BEGIN
    THROW 50001, N'dbo.annotation_pages does not exist. Initialize annotation tables first.', 1;
END;
GO

IF OBJECT_ID(N'dbo.ocr_runs', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ocr_runs
    (
        id                 NVARCHAR(36)  NOT NULL,
        page_id            NVARCHAR(36)  NOT NULL,
        status             NVARCHAR(20)  NOT NULL,
        is_active          BIT           NOT NULL
            CONSTRAINT DF_ocr_runs_is_active DEFAULT (1),
        model_name         NVARCHAR(255) NOT NULL,
        model_version      NVARCHAR(100) NOT NULL,
        device             NVARCHAR(32)  NOT NULL,
        processing_time_ms INT           NULL,
        error_message      NVARCHAR(MAX) NULL,
        created_at         DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_ocr_runs_created_at DEFAULT SYSUTCDATETIME(),
        updated_at         DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_ocr_runs_updated_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_ocr_runs PRIMARY KEY (id),
        CONSTRAINT FK_ocr_runs_page FOREIGN KEY (page_id)
            REFERENCES dbo.annotation_pages(id) ON DELETE CASCADE,
        CONSTRAINT CK_ocr_runs_status CHECK
            (status IN (N'PROCESSING', N'COMPLETED', N'FAILED')),
        CONSTRAINT CK_ocr_runs_processing_time CHECK
            (processing_time_ms IS NULL OR processing_time_ms >= 0)
    );
END;
GO

IF OBJECT_ID(N'dbo.ocr_tokens', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ocr_tokens
    (
        id           NVARCHAR(36)  NOT NULL,
        ocr_run_id   NVARCHAR(36)  NOT NULL,
        token_index  INT           NOT NULL,
        token_id     NVARCHAR(20)  NOT NULL,
        text         NVARCHAR(MAX) NOT NULL,
        confidence   DECIMAL(9,6)  NOT NULL,
        x1           INT           NOT NULL,
        y1           INT           NOT NULL,
        x2           INT           NOT NULL,
        y2           INT           NOT NULL,
        polygon_json NVARCHAR(MAX) NULL,
        include_in_extraction BIT NOT NULL
            CONSTRAINT DF_ocr_tokens_include DEFAULT (1),
        ignored_reason NVARCHAR(64) NULL,
        created_at   DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_ocr_tokens_created_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_ocr_tokens PRIMARY KEY (id),
        CONSTRAINT FK_ocr_tokens_run FOREIGN KEY (ocr_run_id)
            REFERENCES dbo.ocr_runs(id) ON DELETE CASCADE,
        CONSTRAINT UQ_ocr_tokens_run_index UNIQUE (ocr_run_id, token_index),
        CONSTRAINT UQ_ocr_tokens_run_token_id UNIQUE (ocr_run_id, token_id),
        CONSTRAINT CK_ocr_tokens_index CHECK (token_index >= 1),
        CONSTRAINT CK_ocr_tokens_confidence CHECK
            (confidence >= 0 AND confidence <= 1),
        CONSTRAINT CK_ocr_tokens_bbox CHECK
            (x1 >= 0 AND y1 >= 0 AND x2 > x1 AND y2 > y1)
    );
END;
GO

IF COL_LENGTH(N'dbo.ocr_tokens', N'include_in_extraction') IS NULL
BEGIN
    ALTER TABLE dbo.ocr_tokens ADD include_in_extraction BIT NOT NULL
        CONSTRAINT DF_ocr_tokens_include DEFAULT (1) WITH VALUES;
END;
GO

IF COL_LENGTH(N'dbo.ocr_tokens', N'ignored_reason') IS NULL
BEGIN
    ALTER TABLE dbo.ocr_tokens ADD ignored_reason NVARCHAR(64) NULL;
END;
GO

IF OBJECT_ID(N'dbo.ocr_layout_regions', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ocr_layout_regions
    (
        id           NVARCHAR(36) NOT NULL,
        ocr_run_id   NVARCHAR(36) NOT NULL,
        region_index INT          NOT NULL,
        region_type  NVARCHAR(32) NOT NULL,
        confidence   DECIMAL(9,6) NOT NULL,
        x1           INT          NOT NULL,
        y1           INT          NOT NULL,
        x2           INT          NOT NULL,
        y2           INT          NOT NULL,
        created_at   DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_ocr_layout_regions_created_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_ocr_layout_regions PRIMARY KEY (id),
        CONSTRAINT FK_ocr_layout_regions_run FOREIGN KEY (ocr_run_id)
            REFERENCES dbo.ocr_runs(id) ON DELETE CASCADE,
        CONSTRAINT UQ_ocr_layout_regions_run_index UNIQUE (ocr_run_id, region_index),
        CONSTRAINT CK_ocr_layout_regions_index CHECK (region_index >= 1),
        CONSTRAINT CK_ocr_layout_regions_confidence CHECK
            (confidence >= 0 AND confidence <= 1),
        CONSTRAINT CK_ocr_layout_regions_bbox CHECK
            (x1 >= 0 AND y1 >= 0 AND x2 > x1 AND y2 > y1)
    );
END;
GO

IF OBJECT_ID(N'dbo.ocr_reviews', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ocr_reviews
    (
        id               NVARCHAR(36)   NOT NULL,
        ocr_run_id       NVARCHAR(36)   NOT NULL,
        parent_review_id NVARCHAR(36)   NULL,
        status           NVARCHAR(20)   NOT NULL
            CONSTRAINT DF_ocr_reviews_status DEFAULT N'REVIEWED',
        is_active        BIT            NOT NULL
            CONSTRAINT DF_ocr_reviews_is_active DEFAULT (1),
        include_in_llm   BIT            NOT NULL
            CONSTRAINT DF_ocr_reviews_include_in_llm DEFAULT (1),
        reviewed_by      NVARCHAR(100)  NOT NULL,
        review_note      NVARCHAR(MAX)  NULL,
        reviewed_at      DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_ocr_reviews_reviewed_at DEFAULT SYSUTCDATETIME(),
        created_at       DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_ocr_reviews_created_at DEFAULT SYSUTCDATETIME(),
        updated_at       DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_ocr_reviews_updated_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_ocr_reviews PRIMARY KEY (id),
        CONSTRAINT FK_ocr_reviews_run FOREIGN KEY (ocr_run_id)
            REFERENCES dbo.ocr_runs(id) ON DELETE CASCADE,
        CONSTRAINT FK_ocr_reviews_parent FOREIGN KEY (parent_review_id)
            REFERENCES dbo.ocr_reviews(id),
        CONSTRAINT CK_ocr_reviews_status CHECK (status IN (N'REVIEWED'))
    );
END;
GO

IF COL_LENGTH(N'dbo.ocr_reviews', N'include_in_llm') IS NULL
BEGIN
    ALTER TABLE dbo.ocr_reviews ADD include_in_llm BIT NOT NULL
        CONSTRAINT DF_ocr_reviews_include_in_llm DEFAULT (1) WITH VALUES;
END;
GO

IF OBJECT_ID(N'dbo.ocr_review_tokens', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ocr_review_tokens
    (
        id               NVARCHAR(36)  NOT NULL,
        review_id        NVARCHAR(36)  NOT NULL,
        source_token_id  NVARCHAR(36)  NULL,
        token_order      INT           NOT NULL,
        reviewed_text    NVARCHAR(MAX) NOT NULL,
        x1               INT           NOT NULL,
        y1               INT           NOT NULL,
        x2               INT           NOT NULL,
        y2               INT           NOT NULL,
        include_in_extraction BIT      NOT NULL
            CONSTRAINT DF_ocr_review_tokens_include DEFAULT (1),
        ignored_reason   NVARCHAR(64)  NULL,
        created_at       DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_ocr_review_tokens_created_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_ocr_review_tokens PRIMARY KEY (id),
        CONSTRAINT FK_ocr_review_tokens_review FOREIGN KEY (review_id)
            REFERENCES dbo.ocr_reviews(id) ON DELETE CASCADE,
        CONSTRAINT FK_ocr_review_tokens_source FOREIGN KEY (source_token_id)
            REFERENCES dbo.ocr_tokens(id),
        CONSTRAINT UQ_ocr_review_tokens_order UNIQUE (review_id, token_order),
        CONSTRAINT CK_ocr_review_tokens_order CHECK (token_order >= 1),
        CONSTRAINT CK_ocr_review_tokens_bbox CHECK
            (x1 >= 0 AND y1 >= 0 AND x2 > x1 AND y2 > y1)
    );
END;
GO

IF COL_LENGTH(N'dbo.ocr_review_tokens', N'include_in_extraction') IS NULL
BEGIN
    ALTER TABLE dbo.ocr_review_tokens ADD include_in_extraction BIT NOT NULL
        CONSTRAINT DF_ocr_review_tokens_include DEFAULT (1) WITH VALUES;
END;
GO

IF COL_LENGTH(N'dbo.ocr_review_tokens', N'ignored_reason') IS NULL
BEGIN
    ALTER TABLE dbo.ocr_review_tokens ADD ignored_reason NVARCHAR(64) NULL;
END;
GO

IF OBJECT_ID(N'dbo.ocr_review_regions', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ocr_review_regions
    (
        id               NVARCHAR(36) NOT NULL,
        review_id        NVARCHAR(36) NOT NULL,
        source_region_id NVARCHAR(36) NULL,
        region_order     INT          NOT NULL,
        region_type      NVARCHAR(32) NOT NULL,
        x1               INT          NOT NULL,
        y1               INT          NOT NULL,
        x2               INT          NOT NULL,
        y2               INT          NOT NULL,
        created_at       DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_ocr_review_regions_created_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_ocr_review_regions PRIMARY KEY (id),
        CONSTRAINT FK_ocr_review_regions_review FOREIGN KEY (review_id)
            REFERENCES dbo.ocr_reviews(id) ON DELETE CASCADE,
        CONSTRAINT FK_ocr_review_regions_source FOREIGN KEY (source_region_id)
            REFERENCES dbo.ocr_layout_regions(id),
        CONSTRAINT UQ_ocr_review_regions_order UNIQUE (review_id, region_order),
        CONSTRAINT CK_ocr_review_regions_order CHECK (region_order >= 1),
        CONSTRAINT CK_ocr_review_regions_bbox CHECK
            (x1 >= 0 AND y1 >= 0 AND x2 > x1 AND y2 > y1)
    );
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'UX_ocr_reviews_active_run'
      AND object_id = OBJECT_ID(N'dbo.ocr_reviews')
)
BEGIN
    CREATE UNIQUE INDEX UX_ocr_reviews_active_run
        ON dbo.ocr_reviews (ocr_run_id)
        WHERE is_active = 1;
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_ocr_review_tokens_source'
      AND object_id = OBJECT_ID(N'dbo.ocr_review_tokens')
)
BEGIN
    CREATE INDEX IX_ocr_review_tokens_source
        ON dbo.ocr_review_tokens (source_token_id)
        INCLUDE (review_id, token_order);
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'UX_ocr_runs_page_active'
      AND object_id = OBJECT_ID(N'dbo.ocr_runs')
)
BEGIN
    CREATE UNIQUE INDEX UX_ocr_runs_page_active
        ON dbo.ocr_runs (page_id)
        WHERE is_active = 1;
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_ocr_runs_training_export'
      AND object_id = OBJECT_ID(N'dbo.ocr_runs')
)
BEGIN
    CREATE INDEX IX_ocr_runs_training_export
        ON dbo.ocr_runs (status, is_active, created_at)
        INCLUDE (page_id, model_name, model_version, device);
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_ocr_tokens_run_order'
      AND object_id = OBJECT_ID(N'dbo.ocr_tokens')
)
BEGIN
    CREATE INDEX IX_ocr_tokens_run_order
        ON dbo.ocr_tokens (ocr_run_id, token_index)
        INCLUDE (token_id, confidence, x1, y1, x2, y2);
END;
GO

PRINT N'OCR grounding schema is ready.';
GO


/*
Create the SQL Server schema used to review table detection and table structure.

This script is idempotent: existing tables and indexes are left untouched. It
must be executed against the same database that already contains
dbo.annotation_pages.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

IF OBJECT_ID(N'dbo.annotation_pages', N'U') IS NULL
BEGIN
    THROW 50001, N'dbo.annotation_pages does not exist. Initialize the annotation database first.', 1;
END;
GO

/* One row per model prediction or human-reviewed annotation version. */
IF OBJECT_ID(N'dbo.table_annotations', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.table_annotations
    (
        id                   NVARCHAR(36)  NOT NULL,
        page_id              NVARCHAR(36)  NOT NULL,
        parent_annotation_id NVARCHAR(36)  NULL,
        source               NVARCHAR(16)  NOT NULL,
        status               NVARCHAR(20)  NOT NULL,
        is_active            BIT           NOT NULL
            CONSTRAINT DF_table_annotations_is_active DEFAULT (1),
        model_name           NVARCHAR(255) NULL,
        model_version        NVARCHAR(100) NULL,
        processing_time_ms   INT           NULL,
        error_message        NVARCHAR(MAX) NULL,
        reviewed_by          NVARCHAR(100) NULL,
        review_note          NVARCHAR(MAX) NULL,
        reviewed_at          DATETIMEOFFSET(7) NULL,
        created_at           DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_table_annotations_created_at DEFAULT SYSUTCDATETIME(),
        updated_at           DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_table_annotations_updated_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_table_annotations PRIMARY KEY (id),
        CONSTRAINT FK_table_annotations_page FOREIGN KEY (page_id)
            REFERENCES dbo.annotation_pages(id) ON DELETE CASCADE,
        CONSTRAINT FK_table_annotations_parent FOREIGN KEY (parent_annotation_id)
            REFERENCES dbo.table_annotations(id),
        CONSTRAINT CK_table_annotations_source CHECK
            (source IN (N'MODEL', N'HUMAN')),
        CONSTRAINT CK_table_annotations_status CHECK
            (status IN (N'PENDING', N'PROCESSING', N'PREDICTED', N'REVIEWED', N'FAILED')),
        CONSTRAINT CK_table_annotations_processing_time CHECK
            (processing_time_ms IS NULL OR processing_time_ms >= 0),
        CONSTRAINT CK_table_annotations_parent CHECK
            (parent_annotation_id IS NULL OR parent_annotation_id <> id)
    );

    PRINT N'Created table dbo.table_annotations.';
END
ELSE
BEGIN
    PRINT N'Table dbo.table_annotations already exists.';
END;
GO

/* One page may contain more than one table. Coordinates use original page pixels. */
IF OBJECT_ID(N'dbo.table_regions', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.table_regions
    (
        id            NVARCHAR(36) NOT NULL,
        annotation_id NVARCHAR(36) NOT NULL,
        table_index   INT          NOT NULL,
        x1            INT          NOT NULL,
        y1            INT          NOT NULL,
        x2            INT          NOT NULL,
        y2            INT          NOT NULL,
        confidence    DECIMAL(9,6) NULL,
        created_at    DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_table_regions_created_at DEFAULT SYSUTCDATETIME(),
        updated_at    DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_table_regions_updated_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_table_regions PRIMARY KEY (id),
        CONSTRAINT FK_table_regions_annotation FOREIGN KEY (annotation_id)
            REFERENCES dbo.table_annotations(id) ON DELETE CASCADE,
        CONSTRAINT UQ_table_regions_annotation_index UNIQUE (annotation_id, table_index),
        CONSTRAINT CK_table_regions_index CHECK (table_index >= 0),
        CONSTRAINT CK_table_regions_bbox CHECK
            (x1 >= 0 AND y1 >= 0 AND x2 > x1 AND y2 > y1),
        CONSTRAINT CK_table_regions_confidence CHECK
            (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
    );

    PRINT N'Created table dbo.table_regions.';
END
ELSE
BEGIN
    PRINT N'Table dbo.table_regions already exists.';
END;
GO

/* Rows, columns and cells produced by the structure model or human reviewer. */
IF OBJECT_ID(N'dbo.table_structure_elements', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.table_structure_elements
    (
        id              NVARCHAR(36) NOT NULL,
        table_region_id NVARCHAR(36) NOT NULL,
        element_type    NVARCHAR(20) NOT NULL,
        row_index       INT          NULL,
        column_index    INT          NULL,
        row_span        INT          NOT NULL
            CONSTRAINT DF_table_structure_elements_row_span DEFAULT (1),
        column_span     INT          NOT NULL
            CONSTRAINT DF_table_structure_elements_column_span DEFAULT (1),
        x1              INT          NOT NULL,
        y1              INT          NOT NULL,
        x2              INT          NOT NULL,
        y2              INT          NOT NULL,
        confidence      DECIMAL(9,6) NULL,
        created_at      DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_table_structure_elements_created_at DEFAULT SYSUTCDATETIME(),
        updated_at      DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_table_structure_elements_updated_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_table_structure_elements PRIMARY KEY (id),
        CONSTRAINT FK_table_structure_elements_region FOREIGN KEY (table_region_id)
            REFERENCES dbo.table_regions(id) ON DELETE CASCADE,
        CONSTRAINT CK_table_structure_elements_type CHECK
            (element_type IN (N'HEADER_ROW', N'DATA_ROW', N'COLUMN', N'CELL')),
        CONSTRAINT CK_table_structure_elements_indexes CHECK
        (
            (element_type IN (N'HEADER_ROW', N'DATA_ROW')
                AND row_index IS NOT NULL AND row_index >= 0
                AND column_index IS NULL)
            OR
            (element_type = N'COLUMN'
                AND row_index IS NULL
                AND column_index IS NOT NULL AND column_index >= 0)
            OR
            (element_type = N'CELL'
                AND row_index IS NOT NULL AND row_index >= 0
                AND column_index IS NOT NULL AND column_index >= 0)
        ),
        CONSTRAINT CK_table_structure_elements_span CHECK
            (row_span >= 1 AND column_span >= 1),
        CONSTRAINT CK_table_structure_elements_bbox CHECK
            (x1 >= 0 AND y1 >= 0 AND x2 > x1 AND y2 > y1),
        CONSTRAINT CK_table_structure_elements_confidence CHECK
            (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
    );

    PRINT N'Created table dbo.table_structure_elements.';
END
ELSE
BEGIN
    PRINT N'Table dbo.table_structure_elements already exists.';
END;
GO

/* Exactly one current annotation version is allowed per page. */
IF NOT EXISTS
(
    SELECT 1
    FROM sys.indexes
    WHERE name = N'UX_table_annotations_page_active'
      AND object_id = OBJECT_ID(N'dbo.table_annotations')
)
BEGIN
    CREATE UNIQUE INDEX UX_table_annotations_page_active
        ON dbo.table_annotations (page_id)
        WHERE is_active = 1;
END;
GO

IF NOT EXISTS
(
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_table_annotations_page_status'
      AND object_id = OBJECT_ID(N'dbo.table_annotations')
)
BEGIN
    CREATE INDEX IX_table_annotations_page_status
        ON dbo.table_annotations (page_id, status);
END;
GO

IF NOT EXISTS
(
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_table_annotations_review_queue'
      AND object_id = OBJECT_ID(N'dbo.table_annotations')
)
BEGIN
    CREATE INDEX IX_table_annotations_review_queue
        ON dbo.table_annotations (status, is_active, created_at)
        INCLUDE (page_id, source, model_name, model_version);
END;
GO

IF NOT EXISTS
(
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_table_regions_annotation'
      AND object_id = OBJECT_ID(N'dbo.table_regions')
)
BEGIN
    CREATE INDEX IX_table_regions_annotation
        ON dbo.table_regions (annotation_id, table_index);
END;
GO

IF NOT EXISTS
(
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_table_structure_elements_region_type'
      AND object_id = OBJECT_ID(N'dbo.table_structure_elements')
)
BEGIN
    CREATE INDEX IX_table_structure_elements_region_type
        ON dbo.table_structure_elements (table_region_id, element_type, row_index, column_index);
END;
GO

PRINT N'Table annotation schema is ready.';
GO

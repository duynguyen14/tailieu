/* Idempotent SQL Server schema for 19-field visa application form bbox review. */
SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

IF OBJECT_ID(N'dbo.annotation_pages', N'U') IS NULL
BEGIN
    THROW 50001, N'dbo.annotation_pages does not exist. Initialize the annotation database first.', 1;
END;
GO

IF OBJECT_ID(N'dbo.visa_form_annotations', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.visa_form_annotations
    (
        id                   VARCHAR(36)   NOT NULL,
        page_id              VARCHAR(36)   NOT NULL,
        parent_annotation_id VARCHAR(36)   NULL,
        source               NVARCHAR(16)  NOT NULL,
        status               NVARCHAR(20)  NOT NULL,
        template_version     NVARCHAR(100) NOT NULL,
        is_active            BIT NOT NULL CONSTRAINT DF_visa_form_annotations_active DEFAULT (1),
        image_width          INT NOT NULL,
        image_height         INT NOT NULL,
        reviewed_by          NVARCHAR(100) NULL,
        review_note          NVARCHAR(MAX) NULL,
        reviewed_at          DATETIMEOFFSET(7) NULL,
        created_at           DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_visa_form_annotations_created DEFAULT SYSUTCDATETIME(),
        updated_at           DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_visa_form_annotations_updated DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_visa_form_annotations PRIMARY KEY (id),
        CONSTRAINT FK_visa_form_annotations_page FOREIGN KEY (page_id)
            REFERENCES dbo.annotation_pages(id) ON DELETE CASCADE,
        CONSTRAINT FK_visa_form_annotations_parent FOREIGN KEY (parent_annotation_id)
            REFERENCES dbo.visa_form_annotations(id),
        CONSTRAINT CK_visa_form_annotations_source CHECK (source IN (N'TEMPLATE', N'HUMAN')),
        CONSTRAINT CK_visa_form_annotations_status CHECK
            (status IN (N'PENDING', N'REVIEWED', N'REJECTED')),
        CONSTRAINT CK_visa_form_annotations_dimensions CHECK
            (image_width > 0 AND image_height > 0),
        CONSTRAINT CK_visa_form_annotations_parent CHECK
            (parent_annotation_id IS NULL OR parent_annotation_id <> id)
    );
END;
GO

IF OBJECT_ID(N'dbo.visa_form_field_annotations', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.visa_form_field_annotations
    (
        id            VARCHAR(36) NOT NULL,
        annotation_id VARCHAR(36) NOT NULL,
        field_code    NVARCHAR(64) NOT NULL,
        field_order   INT NOT NULL,
        value_text    NVARCHAR(MAX) NULL,
        x1 INT NOT NULL,
        y1 INT NOT NULL,
        x2 INT NOT NULL,
        y2 INT NOT NULL,
        nx1 INT NOT NULL,
        ny1 INT NOT NULL,
        nx2 INT NOT NULL,
        ny2 INT NOT NULL,
        created_at DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_visa_form_fields_created DEFAULT SYSUTCDATETIME(),
        updated_at DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_visa_form_fields_updated DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_visa_form_field_annotations PRIMARY KEY (id),
        CONSTRAINT FK_visa_form_fields_annotation FOREIGN KEY (annotation_id)
            REFERENCES dbo.visa_form_annotations(id) ON DELETE CASCADE,
        CONSTRAINT UQ_visa_form_fields_code UNIQUE (annotation_id, field_code),
        CONSTRAINT UQ_visa_form_fields_order UNIQUE (annotation_id, field_order),
        CONSTRAINT CK_visa_form_fields_order CHECK (field_order >= 1),
        CONSTRAINT CK_visa_form_fields_bbox CHECK
            (x1 >= 0 AND y1 >= 0 AND x2 > x1 AND y2 > y1),
        CONSTRAINT CK_visa_form_fields_normalized_bbox CHECK
            (nx1 >= 0 AND ny1 >= 0 AND nx2 <= 1000 AND ny2 <= 1000
             AND nx2 > nx1 AND ny2 > ny1)
    );
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'UX_visa_form_annotations_page_active'
      AND object_id = OBJECT_ID(N'dbo.visa_form_annotations')
)
BEGIN
    CREATE UNIQUE INDEX UX_visa_form_annotations_page_active
        ON dbo.visa_form_annotations(page_id) WHERE is_active = 1;
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_visa_form_annotations_queue'
      AND object_id = OBJECT_ID(N'dbo.visa_form_annotations')
)
BEGIN
    CREATE INDEX IX_visa_form_annotations_queue
        ON dbo.visa_form_annotations(status, is_active, created_at DESC)
        INCLUDE (page_id, template_version, source);
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_visa_form_fields_annotation_order'
      AND object_id = OBJECT_ID(N'dbo.visa_form_field_annotations')
)
BEGIN
    CREATE INDEX IX_visa_form_fields_annotation_order
        ON dbo.visa_form_field_annotations(annotation_id, field_order);
END;
GO

PRINT N'Visa form annotation schema is ready.';
GO

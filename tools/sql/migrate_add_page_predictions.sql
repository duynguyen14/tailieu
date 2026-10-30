/* Add model-suggestion metadata to the existing page review table. */
SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

IF OBJECT_ID(N'dbo.annotation_pages', N'U') IS NULL
BEGIN
    THROW 50002, N'dbo.annotation_pages does not exist. Initialize the annotation database first.', 1;
END;
GO

IF COL_LENGTH(N'dbo.annotation_pages', N'model_predicted_label') IS NULL
BEGIN
    ALTER TABLE dbo.annotation_pages ADD model_predicted_label NVARCHAR(64) NULL;
END;
GO

IF COL_LENGTH(N'dbo.annotation_pages', N'model_prediction_confidence') IS NULL
BEGIN
    ALTER TABLE dbo.annotation_pages ADD model_prediction_confidence DECIMAL(9,6) NULL;
END;
GO

IF COL_LENGTH(N'dbo.annotation_pages', N'model_prediction_topk_json') IS NULL
BEGIN
    ALTER TABLE dbo.annotation_pages ADD model_prediction_topk_json NVARCHAR(MAX) NULL;
END;
GO

IF COL_LENGTH(N'dbo.annotation_pages', N'model_prediction_version') IS NULL
BEGIN
    ALTER TABLE dbo.annotation_pages ADD model_prediction_version NVARCHAR(100) NULL;
END;
GO

IF COL_LENGTH(N'dbo.annotation_pages', N'model_predicted_at') IS NULL
BEGIN
    ALTER TABLE dbo.annotation_pages ADD model_predicted_at DATETIMEOFFSET(7) NULL;
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.check_constraints
    WHERE name = N'CK_annotation_pages_prediction_label'
      AND parent_object_id = OBJECT_ID(N'dbo.annotation_pages')
)
BEGIN
    ALTER TABLE dbo.annotation_pages WITH CHECK
        ADD CONSTRAINT CK_annotation_pages_prediction_label CHECK
        (
            model_predicted_label IS NULL OR model_predicted_label IN
            (
                N'DIPLOMATIC_NOTE', N'APPLICANT_TABLE', N'APPLICANT_LIST',
                N'VISA_APPLICATION_FORM', N'TEMPORARY_RESIDENCE_EXTENSION_FORM',
                N'PASSPORT_FRONT', N'VISA_STICKER',
                N'VISA_REQUEST_OFFICIAL_LETTER', N'OTHER'
            )
        );
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.check_constraints
    WHERE name = N'CK_annotation_pages_prediction_confidence'
      AND parent_object_id = OBJECT_ID(N'dbo.annotation_pages')
)
BEGIN
    ALTER TABLE dbo.annotation_pages WITH CHECK
        ADD CONSTRAINT CK_annotation_pages_prediction_confidence CHECK
        (
            model_prediction_confidence IS NULL
            OR (model_prediction_confidence >= 0 AND model_prediction_confidence <= 1)
        );
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.check_constraints
    WHERE name = N'CK_annotation_pages_prediction_topk_json'
      AND parent_object_id = OBJECT_ID(N'dbo.annotation_pages')
)
BEGIN
    ALTER TABLE dbo.annotation_pages WITH CHECK
        ADD CONSTRAINT CK_annotation_pages_prediction_topk_json CHECK
        (
            model_prediction_topk_json IS NULL
            OR ISJSON(model_prediction_topk_json) = 1
        );
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_annotation_pages_prediction_queue'
      AND object_id = OBJECT_ID(N'dbo.annotation_pages')
)
BEGIN
    CREATE INDEX IX_annotation_pages_prediction_queue
        ON dbo.annotation_pages (review_status, model_prediction_version, model_predicted_at)
        INCLUDE (label, model_predicted_label, model_prediction_confidence);
END;
GO

PRINT N'Page prediction columns are ready.';
GO

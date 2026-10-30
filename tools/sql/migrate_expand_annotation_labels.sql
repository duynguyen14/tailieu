USE [document_annotation];
GO

/* Preserve all rows; only replace the allowed-value check constraint. */
IF EXISTS
(
    SELECT 1
    FROM sys.check_constraints
    WHERE name = N'CK_annotation_pages_label'
      AND parent_object_id = OBJECT_ID(N'dbo.annotation_pages')
)
BEGIN
    ALTER TABLE dbo.annotation_pages
        DROP CONSTRAINT CK_annotation_pages_label;
END;
GO

ALTER TABLE dbo.annotation_pages WITH CHECK
    ADD CONSTRAINT CK_annotation_pages_label CHECK
    (
        label IS NULL OR label IN
        (
            N'DIPLOMATIC_NOTE',
            N'APPLICANT_TABLE',
            N'APPLICANT_LIST',
            N'VISA_APPLICATION_FORM',
            N'TEMPORARY_RESIDENCE_EXTENSION_FORM',
            N'PASSPORT_FRONT',
            N'PASSPORT_BACK',
            N'VISA_STICKER',
            N'OTHER'
        )
    );
GO

ALTER TABLE dbo.annotation_pages
    CHECK CONSTRAINT CK_annotation_pages_label;
GO

PRINT N'Annotation label constraint now accepts the expanded label set.';
GO

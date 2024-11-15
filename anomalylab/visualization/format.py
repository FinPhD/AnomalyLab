from anomalylab.utils.imports import *


@dataclass
class FormatExcel:
    """Class to format Excel files by adjusting cell alignment and borders.

    Attributes:
        path (str): The directory or file path of the Excel file(s) to format.
    """

    path: str

    def load_workbook(self, file_path: str):
        """Loads an Excel workbook from the specified file path.

        Args:
            file_path (str): The path to the Excel file to load.
        """
        self.wb = load_workbook(file_path)
        self.file_path = file_path

    def align(self):
        """Aligns the text in the Excel cells.

        - Aligns text in the first column to the left.
        - Aligns text in all other columns to the center.
        """
        for ws in self.wb.worksheets:
            for row in ws.iter_rows(
                min_row=1, max_row=ws.max_row, min_col=1, max_col=ws.max_column
            ):
                for cell in row:
                    if cell.column == 1:
                        cell.alignment = Alignment(horizontal="left", vertical="center")
                    else:
                        cell.alignment = Alignment(
                            horizontal="center", vertical="center"
                        )

    def line(self):
        """Applies borders to the cells in the Excel workbook.

        - Creates thin borders for the top of the first row and bottom of the last row.
        - Creates a thick border for the bottom of the first row.
        """
        thin = Side(border_style="thin", color="000000")
        thick = Side(border_style="thick", color="000000")

        for ws in self.wb.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    cell.border = Border()

            max_row = ws.max_row

            for cell in ws[1]:
                cell.border = Border(top=thin, bottom=thin)

            for cell in ws[max_row]:
                cell.border = Border(bottom=thin)

    def save(self):
        """Saves the currently loaded workbook to its file path."""
        self.wb.save(self.file_path)

    def process(self):
        """Processes and formats Excel files.

        - If the provided path is a directory, it formats all Excel files in that directory.
        - If the provided path is a file, it formats that specific Excel file.
        """
        if os.path.isdir(self.path):
            for file in glob(os.path.join(self.path, "*.xlsx")):
                self.load_workbook(file)
                self.align()
                self.line()
                self.save()
        else:
            self.load_workbook(self.path)
            self.align()
            self.line()
            self.save()


if __name__ == "__main__":
    path = "..."
    excel_formatter = FormatExcel(path)
    excel_formatter.process()

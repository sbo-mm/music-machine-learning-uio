class MgProgressbar():
    """
    Calls in a loop to create terminal progress bar.
    """

    def __init__(
            self,
            total=100,
            time_limit=0.5,
            prefix='Progress',
            suffix='Complete',
            decimals=1,
            length=40,
            fill='█'):
        """
        Initialize the MgProgressbar object.
        Args:
            total (int, optional): Total iterations. Defaults to 100.
            time_limit (float, optional): The minimum refresh rate of the progressbar in seconds. Defaults to 0.5.
            prefix (str, optional): Prefix string. Defaults to 'Progress'.
            suffix (str, optional): Suffix string. Defaults to 'Complete'.
            decimals (int, optional): Positive number of decimals in process percent. Defaults to 1.
            length (int, optional): Character length of the status bar. Defaults to 40.
            fill (str, optional): Bar fill character. Defaults to '█'.
        """

        self.total = total - 1
        self.time_limit = time_limit
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.now = self.get_now()
        self.finished = False
        self.could_not_get_terminal_window = False
        self.tw_width = 0
        self.tw_height = 0
        self.display_only_percent = False

    def get_now(self):
        """
        Gets the current time.
        Returns:
            datetime.datetime.timestamp: The current time.
        """
        from datetime import datetime
        return datetime.timestamp(datetime.now())

    def over_time_limit(self):
        """
        Checks if we should redraw the progress bar at this moment.
        Returns:
            bool: True if equal or more time has passed than `self.time_limit` since the last redraw.
        """
        callback_time = self.get_now()
        return callback_time - self.now >= self.time_limit

    def adjust_printlength(self):
        if self.tw_width <= 0:
            return
        elif self.could_not_get_terminal_window:
            return
        else:
            current_length = len(self.prefix) + self.length + \
                self.decimals + len(self.suffix) + 10
            if current_length > self.tw_width:
                diff = current_length - self.tw_width
                if diff < self.length:
                    self.length -= diff
                else:  # remove suffix
                    current_length = current_length - len(self.suffix)
                    diff = current_length - self.tw_width
                    if diff <= 0:
                        self.suffix = ""
                    elif diff < self.length:
                        self.suffix = ""
                        self.length -= diff
                    else:  # remove prefix
                        current_length = current_length - len(self.prefix)
                        diff = current_length - self.tw_width
                        if diff <= 0:
                            self.suffix = ""
                            self.prefix = ""
                        elif diff < self.length:
                            self.suffix = ""
                            self.prefix = ""
                            self.length -= diff
                        else:  # display only percent
                            self.display_only_percent = True

    def progress(self, iteration):
        """
        Progresses the progress bar to the next step.
        Args:
            iteration (float): The current iteration. For example, the 57th out of 100 steps, or 12.3s out of the total 60s.
        """
        if self.finished:
            return
        import sys
        import shutil

        if not self.could_not_get_terminal_window:
            self.tw_width, self.tw_height = shutil.get_terminal_size((0, 0))
            if self.tw_width + self.tw_height == 0:
                self.could_not_get_terminal_window = True
            else:
                self.adjust_printlength()

        capped_iteration = iteration if iteration <= self.total else self.total
        # Print New Line on Complete
        if iteration >= self.total:
            self.finished = True
            percent = ("{0:." + str(self.decimals) + "f}").format(100)
            filledLength = int(round(self.length))
            bar = self.fill * filledLength
            sys.stdout.flush()
            if self.display_only_percent:
                sys.stdout.write('\r%s' % (percent))
            else:
                sys.stdout.write('\r%s |%s| %s%% %s' %
                                 (self.prefix, bar, percent, self.suffix))
            print()
        elif self.over_time_limit():
            self.now = self.get_now()
            percent = ("{0:." + str(self.decimals) + "f}").format(100 *
                                                                  (capped_iteration / float(self.total)))
            filledLength = int(self.length * capped_iteration // self.total)
            bar = self.fill * filledLength + '-' * (self.length - filledLength)
            sys.stdout.flush()
            if self.display_only_percent:
                sys.stdout.write('\r%s' % (percent))
            else:
                sys.stdout.write('\r%s |%s| %s%% %s' %
                                 (self.prefix, bar, percent, self.suffix))
        else:
            return

    def __repr__(self):
        return "MgProgressbar"